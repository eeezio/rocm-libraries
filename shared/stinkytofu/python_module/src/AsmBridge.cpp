// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "stinkytofu/analysis/asm/AsmVerifierPass.hpp"
#include "stinkytofu/core/Function.hpp"
#include "stinkytofu/hardware/ArchHelper.hpp"
#include "stinkytofu/hardware/ComgrProbe.hpp"
#include "stinkytofu/hardware/HwReg.hpp"
#include "stinkytofu/ir/asm/StinkyAsmDirectives.hpp"
#include "stinkytofu/ir/asm/StinkyAsmIR.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"

namespace nb = nanobind;
using namespace stinkytofu;

namespace {

[[noreturn]] void fail(size_t itemIndex, const std::string& message) {
    throw nb::value_error(("emit_asm item[" + std::to_string(itemIndex) + "]: " + message).c_str());
}

nb::handle find(const nb::dict& record, const char* key) {
    PyObject* value = PyDict_GetItemString(record.ptr(), key);
    return value ? nb::handle(value) : nb::handle();
}

nb::handle require(const nb::dict& record, const char* key, size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) fail(itemIndex, std::string("missing required field '") + key + "'");
    return value;
}

std::string requireString(const nb::dict& record, const char* key, size_t itemIndex) {
    nb::handle value = require(record, key, itemIndex);
    if (!PyUnicode_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be a string");
    return nb::cast<std::string>(value);
}

int64_t requireInteger(nb::handle value, const std::string& field, size_t itemIndex) {
    if (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
        fail(itemIndex, "field '" + field + "' must be an integer");
    try {
        return nb::cast<int64_t>(value);
    } catch (const nb::cast_error&) {
        fail(itemIndex, "field '" + field + "' is outside the signed 64-bit range");
    }
}

bool optionalBool(const nb::dict& record, const char* key, bool defaultValue, size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) return defaultValue;
    if (!PyBool_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be bool");
    return value.ptr() == Py_True;
}

int64_t optionalInteger(const nb::dict& record, const char* key, int64_t defaultValue,
                        size_t itemIndex) {
    nb::handle value = find(record, key);
    return value.is_valid() ? requireInteger(value, key, itemIndex) : defaultValue;
}

std::vector<int> optionalIntegerVector(const nb::dict& record, const char* key,
                                       size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) return {};
    if (!PyList_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be a list of integers");
    std::vector<int> result;
    for (nb::handle element : nb::borrow<nb::list>(value)) {
        int64_t parsed = requireInteger(element, key, itemIndex);
        if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max())
            fail(itemIndex, std::string("field '") + key + "' value is outside the int range");
        result.push_back(static_cast<int>(parsed));
    }
    return result;
}

void rejectUnknownKeys(const nb::dict& record, const std::unordered_set<std::string>& allowed,
                       size_t itemIndex) {
    Py_ssize_t pos = 0;
    PyObject* key = nullptr;
    PyObject* value = nullptr;
    while (PyDict_Next(record.ptr(), &pos, &key, &value)) {
        if (!PyUnicode_Check(key)) fail(itemIndex, "record keys must be strings");
        std::string name = nb::cast<std::string>(nb::handle(key));
        if (!allowed.contains(name)) fail(itemIndex, "unknown field '" + name + "'");
    }
}

bool isLabelName(const std::string& name) {
    if (name.empty()) return false;
    auto first = static_cast<unsigned char>(name.front());
    if (!(std::isalpha(first) || name.front() == '_' || name.front() == '.' ||
          name.front() == '$'))
        return false;
    return std::all_of(name.begin() + 1, name.end(), [](char c) {
        auto u = static_cast<unsigned char>(c);
        return std::isalnum(u) || c == '_' || c == '.' || c == '$';
    });
}

bool isSupportedTypedInstruction(const std::string& mnemonic) {
    static const std::unordered_set<std::string> supported = {
        // T00 FMHA typed forms. Packed v_pk_fma_f32/v_pk_mul_f32 remain audited custom words.
        "ds_load_b128", "ds_load_tr16_b128", "ds_store_b128", "ds_store_b32",
        "flat_load_b32", "flat_store_b32", "global_prefetch_b8", "s_add_co_ci_u32",
        "s_add_co_i32", "s_add_co_u32", "s_add_u32",
        "s_addk_co_i32", "s_and_b32", "s_ashr_i32", "s_barrier_signal",
        "s_barrier_wait", "s_bfe_u32", "s_bitcmp1_b32", "s_bitset0_b32",
        "s_bitset1_b32", "s_cselect_b32", "s_cmp_eq_u32", "s_cmp_ge_u32",
        "s_cmp_lt_i32", "s_cmp_lt_u32", "s_code_end", "s_endpgm", "s_getpc_b64",
        "s_lshl2_add_u32", "s_lshl_b32", "s_lshr_b32", "s_load_b32", "s_load_b64",
        "s_max_i32", "s_min_i32", "s_min_u32", "s_mov_b32", "s_mul_hi_u32",
        "s_mul_i32", "s_nop", "s_or_b32", "s_prefetch_inst", "s_setreg_IMM32_b32",
        "s_sub_co_i32", "s_sub_u32", "s_version", "s_wait_alu", "s_wait_dscnt",
        "s_wait_idle", "s_wait_kmcnt", "s_wait_loadcnt", "s_wait_storecnt",
        "s_wait_tensorcnt", "s_waitcnt", "s_xor_b32",
        "tensor_load_to_lds", "tensor_store_from_lds", "v_add_co_ci_u32", "v_add_co_u32",
        "v_add_f32", "v_add_nc_i32", "v_add_nc_u32", "v_add_nc_u64", "v_and_b32",
        "v_cmp_eq_f32", "v_cmp_le_u32", "v_cmp_lt_i32",
        "v_cmp_lt_u32", "v_cndmask_b32", "v_cvt_f32_u32", "v_cvt_pk_bf16_f32",
        "v_cvt_u32_f32", "v_exp_f32", "v_fma_f32", "v_log_f32", "v_lshlrev_b32",
        "v_lshrrev_b32", "v_max3_num_f32", "v_max_num_f32", "v_mov_b32", "v_mov_b64",
        "v_mul_f32", "v_mul_hi_u32", "v_mul_i32_i24", "v_mul_lo_u32",
        "v_mul_u32_u24", "v_nop", "v_permlane16_swap_b32", "v_permlanex16_b32",
        "v_pk_add_f32", "v_pk_mul_f32", "v_rcp_f32", "v_rcp_iflag_f32", "v_readfirstlane_b32",
        "v_readlane_b32", "v_sub_f32", "v_sub_nc_i32", "v_sub_u32", "v_subrev_nc_u32",
        "v_swap_b32", "v_wmma_f32_16x16x32_bf16", "v_writelane_b32",
        // Structural records selected by the target lowering.
        "s_branch", "s_cbranch_execnz", "s_cbranch_execz", "s_cbranch_scc0",
        "s_cbranch_scc1", "s_cbranch_vccnz", "s_cbranch_vccz", "s_set_vgpr_msb",
        // Forms used by the Daedalus MXFP8 decode and FMHA kernels; some reach their
        // exact bits through the "encoding" override below.
        "ds_load_b32", "ds_load_b64", "ds_load_tr8_b64",
        "global_load_async_to_lds_b32", "global_load_async_to_lds_b64",
        "global_load_async_to_lds_b128", "global_store_async_from_lds_b32",
        "global_store_async_from_lds_b64", "global_store_async_from_lds_b128",
        "global_store_b8", "s_cmp_eq_i32", "s_cmp_gt_i32", "s_cselect_b64",
        "s_getreg_b32", "s_lshl_b64", "s_mov_b64", "s_mul_hi_i32",
        "s_wait_asynccnt", "s_wait_xcnt", "v_sub_nc_u32",
        "v_lshl_add_u32", "v_min_u64", "v_cvt_pk_fp8_f32", "v_pk_fma_f32",
        "v_wmma_scale_f32_16x16x128_f8f6f4",
        // Forms used by the Daedalus MXFP8 forward kernel. v_max_u32, v_min_u32,
        // v_xor_b32, v_cmp_gt_i32 and v_cmp_ge_f32 are VOP3-only here (2 sources);
        // their unused SRC2 field needs the "encoding" override below to force 0
        // instead of LLVM's inline-0 (128), matching the SP3-authored custom route.
        "s_ctz_i32_b32", "s_sub_co_ci_u32", "s_cmp_le_i32", "s_lshl1_add_u32",
        "v_max_u32", "v_min_u32", "v_xor_b32", "v_cmp_gt_i32", "v_cmp_ge_f32",
        "ds_load_u16_d16",
    };
    return supported.contains(mnemonic);
}

std::vector<StinkyRegister> parseOperands(const nb::dict& record, const char* key,
                                          size_t itemIndex) {
    nb::handle value = find(record, key);
    if (!value.is_valid()) return {};
    if (!PyList_Check(value.ptr()))
        fail(itemIndex, std::string("field '") + key + "' must be a list of Register values");

    std::vector<StinkyRegister> result;
    nb::list values = nb::borrow<nb::list>(value);
    result.reserve(values.size());
    size_t operandIndex = 0;
    for (nb::handle operand : values) {
        if (!nb::isinstance<StinkyRegister>(operand)) {
            fail(itemIndex, std::string("field '") + key + "'[" +
                                std::to_string(operandIndex) + "] must be a Register");
        }
        result.push_back(nb::cast<StinkyRegister>(operand));
        ++operandIndex;
    }
    return result;
}

void addModifiers(StinkyInstruction& inst, const nb::dict& record, size_t itemIndex) {
    nb::handle value = find(record, "modifiers");
    if (!value.is_valid()) return;
    if (!PyDict_Check(value.ptr())) fail(itemIndex, "field 'modifiers' must be a dict");
    nb::dict modifiers = nb::borrow<nb::dict>(value);
    rejectUnknownKeys(modifiers,
                      {"ds", "global", "smem", "vop3p", "mfma", "wmma_scale", "swaitcnt",
                       "swaittensorcnt", "waitalu", "encoding"},
                      itemIndex);

    if (nb::handle dsValue = find(modifiers, "ds"); dsValue.is_valid()) {
        if (!(inst.is(IF_DSRead) || inst.is(IF_DSStore) || inst.is(IF_DSAtomic)))
            fail(itemIndex, "modifier 'ds' is only valid on DS instructions");
        if (!PyDict_Check(dsValue.ptr())) fail(itemIndex, "modifier 'ds' must be a dict");
        nb::dict ds = nb::borrow<nb::dict>(dsValue);
        rejectUnknownKeys(ds, {"na", "offset", "offset0", "offset1", "gds"}, itemIndex);

        int64_t na = optionalInteger(ds, "na", 1, itemIndex);
        int64_t offset = optionalInteger(ds, "offset", 0, itemIndex);
        int64_t offset0 = optionalInteger(ds, "offset0", 0, itemIndex);
        int64_t offset1 = optionalInteger(ds, "offset1", 0, itemIndex);
        if (na < 1 || na > 2) fail(itemIndex, "DS modifier 'na' must be 1 or 2");
        if (offset < 0 || offset > 65535)
            fail(itemIndex, "DS modifier 'offset' must be in [0, 65535]");
        if (offset0 < 0 || offset0 > 255 || offset1 < 0 || offset1 > 255)
            fail(itemIndex, "DS modifier offset0/offset1 must be in [0, 255]");
        if (na == 1 && (offset0 != 0 || offset1 != 0))
            fail(itemIndex, "DS modifier offset0/offset1 require na=2");
        if (optionalBool(ds, "gds", false, itemIndex))
            fail(itemIndex, "DS modifier gds=True is unsupported on the gfx1250 bridge");
        inst.addModifier(DSModifiers(static_cast<int>(na), static_cast<int>(offset),
                                     static_cast<int>(offset0), static_cast<int>(offset1),
                                     false));
    }

    if (nb::handle value = find(modifiers, "vop3p"); value.is_valid()) {
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'vop3p' must be a dict");
        const MicrocodeFormat microcode = inst.getHwInstDesc()->microcode;
        const bool isVop3p = microcode == MicrocodeFormat::MC_VOP3P ||
                             microcode == MicrocodeFormat::MC_VOP3PX2 ||
                             microcode == MicrocodeFormat::MC_VOP3PX3;
        const bool isVop3 = microcode == MicrocodeFormat::MC_VOP3;
        if (!isVop3p && !isVop3)
            fail(itemIndex, "modifier 'vop3p' is only valid on VOP3P (or VOP3, op_sel-only) "
                             "instructions");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"op_sel", "op_sel_hi", "byte_sel", "neg_lo", "neg_hi"}, itemIndex);
        // Plain VOP3 (not VOP3P) instructions accept only op_sel (e.g. v_cvt_pk_fp8_f32's
        // destination-half select); op_sel_hi/byte_sel/neg_lo/neg_hi are VOP3P-only fields
        // and do not exist in the VOP3 encoding.
        if (!isVop3p) {
            rejectUnknownKeys(data, {"op_sel"}, itemIndex);
        }
        auto opSel = optionalIntegerVector(data, "op_sel", itemIndex);
        auto opSelHi = optionalIntegerVector(data, "op_sel_hi", itemIndex);
        auto byteSel = optionalIntegerVector(data, "byte_sel", itemIndex);
        auto negLo = optionalIntegerVector(data, "neg_lo", itemIndex);
        auto negHi = optionalIntegerVector(data, "neg_hi", itemIndex);
        for (const auto* pair : {&opSel, &opSelHi, &byteSel, &negLo, &negHi})
            if (std::any_of(pair->begin(), pair->end(), [](int bit) { return bit != 0 && bit != 1; }))
                fail(itemIndex, "VOP3P selector fields must contain only 0 or 1");
        inst.addModifier(VOP3PModifiers(opSel, opSelHi, byteSel, negLo, negHi));
    }

    if (nb::handle value = find(modifiers, "wmma_scale"); value.is_valid()) {
        const std::string_view mnemonic(inst.getHwInstDesc()->mnemonic);
        if (!mnemonic.starts_with("v_wmma_scale"))
            fail(itemIndex, "modifier 'wmma_scale' is only valid on v_wmma_scale_* instructions");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'wmma_scale' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data,
                          {"scale_sel_a", "scale_sel_b", "fmt_a", "fmt_b", "scale_fmt_a",
                           "scale_fmt_b"},
                          itemIndex);
        auto parseMatrixFmtKey = [&](const char* key) {
            if (!find(data, key).is_valid()) return MatrixFmt::NONE;
            std::string text = requireString(data, key, itemIndex);
            MatrixFmt fmt = parseMatrixFmt(std::string_view(text));
            if (fmt == MatrixFmt::NONE && text != "MATRIX_FMT_NONE")
                fail(itemIndex, std::string("unknown wmma_scale '") + key + "' value '" + text + "'");
            return fmt;
        };
        auto parseScaleFmt = [&](const char* key) {
            if (!find(data, key).is_valid()) return MatrixScaleFmt::NONE;
            int64_t raw = optionalInteger(data, key, 0, itemIndex);
            if (raw < 0 || raw > 2)
                fail(itemIndex, std::string("wmma_scale '") + key + "' must be in [0, 2]");
            return static_cast<MatrixScaleFmt>(raw);
        };
        MatrixFmtModifiers matrixMod;
        matrixMod.fmtA = parseMatrixFmtKey("fmt_a");
        matrixMod.fmtB = parseMatrixFmtKey("fmt_b");
        matrixMod.scaleFmtA = parseScaleFmt("scale_fmt_a");
        matrixMod.scaleFmtB = parseScaleFmt("scale_fmt_b");
        matrixMod.scaleSelA = static_cast<int>(optionalInteger(data, "scale_sel_a", 0, itemIndex));
        matrixMod.scaleSelB = static_cast<int>(optionalInteger(data, "scale_sel_b", 0, itemIndex));
        if ((matrixMod.scaleSelA & ~1) != 0 || (matrixMod.scaleSelB & ~1) != 0)
            fail(itemIndex, "wmma_scale 'scale_sel_a'/'scale_sel_b' must be 0 or 1");
        inst.addModifier(matrixMod);
    }

    if (nb::handle value = find(modifiers, "mfma"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) !=
            "v_wmma_f32_16x16x32_bf16")
            fail(itemIndex,
                 "modifier 'mfma' is only valid on v_wmma_f32_16x16x32_bf16");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'mfma' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"neg", "reuse_a", "reuse_b"}, itemIndex);
        MFMAModifiers modifier;
        if (optionalBool(data, "neg", false, itemIndex)) {
            modifier.negBits.negLo = {1, 1, 0};
            modifier.negBits.numSrcs = 2;
        }
        modifier.reuseA = optionalBool(data, "reuse_a", false, itemIndex);
        modifier.reuseB = optionalBool(data, "reuse_b", false, itemIndex);
        inst.addModifier(modifier);
    }

    if (nb::handle value = find(modifiers, "smem"); value.is_valid()) {
        const std::string_view mnemonic(inst.getHwInstDesc()->mnemonic);
        if (mnemonic != "s_load_b32" && mnemonic != "s_load_b64")
            fail(itemIndex, "modifier 'smem' is only valid on s_load_b32/s_load_b64");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'smem' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"glc", "nv", "offset"}, itemIndex);
        int64_t offset = optionalInteger(data, "offset", 0, itemIndex);
        if (offset < std::numeric_limits<int>::min() || offset > std::numeric_limits<int>::max())
            fail(itemIndex, "SMEM modifier 'offset' is outside the int range");
        inst.addModifier(SMEMModifiers(optionalBool(data, "glc", false, itemIndex),
                                       optionalBool(data, "nv", false, itemIndex),
                                       static_cast<int>(offset)));
    }

    if (nb::handle value = find(modifiers, "global"); value.is_valid()) {
        // Gated on the VGLOBAL microcode format (table-driven), not a per-mnemonic
        // allowlist: every VGLOBAL-format instruction (global_prefetch_b8,
        // global_store_b8, global_load_async_to_lds_*, global_store_async_from_lds_*,
        // and any future global_* addition) accepts the same offset/scope/temporal_hint
        // triple.
        if (inst.getHwInstDesc()->microcode != MicrocodeFormat::MC_VGLOBAL)
            fail(itemIndex, "modifier 'global' is only valid on VGLOBAL-format instructions");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'global' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"offset", "scope", "temporal_hint"}, itemIndex);
        int64_t offset = optionalInteger(data, "offset", 0, itemIndex);
        if (offset < 0 || offset > std::numeric_limits<int>::max())
            fail(itemIndex, "GLOBAL modifier 'offset' must be a nonnegative int");
        std::string scope = find(data, "scope").is_valid()
                                ? requireString(data, "scope", itemIndex)
                                : "SCOPE_NONE";
        std::string hint = find(data, "temporal_hint").is_valid()
                               ? requireString(data, "temporal_hint", itemIndex)
                               : "TH_NONE";
        MUBUFScope parsedScope = parseMUBUFScope(scope);
        TemporalHint parsedHint = parseTemporalHint(hint);
        if (scope != "SCOPE_NONE" && parsedScope == MUBUFScope::SCOPE_NONE)
            fail(itemIndex, "unknown GLOBAL scope '" + scope + "'");
        if (hint != "TH_NONE" && parsedHint == TemporalHint::TH_NONE)
            fail(itemIndex, "unknown GLOBAL temporal_hint '" + hint + "'");
        // th printing shares TH[2:0] encodings between load and store direction but
        // uses different mnemonic suffixes (e.g. TH_LOAD_LU vs TH_STORE_WB for the
        // same bit pattern), so GLOBALModifiers must know which direction this
        // instruction is. global_store_b8 and global_store_async_from_lds_* are
        // store-direction; global_prefetch_b8 and global_load_async_to_lds_* are
        // load-direction (the latter via IF_GLOBALLoadAsyncToLds, not IF_GLOBALStore).
        bool isStore = isGLOBALStore(inst) || isGlobalStoreAsyncFromLds(inst);
        inst.addModifier(
            GLOBALModifiers(static_cast<int>(offset), parsedHint, parsedScope,
                            /*glc=*/false, /*slc=*/false, /*dlc=*/false, /*lds=*/false, isStore));
    }

    if (nb::handle value = find(modifiers, "swaitcnt"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "s_waitcnt")
            fail(itemIndex, "modifier 'swaitcnt' is only valid on s_waitcnt");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'swaitcnt' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"vlcnt", "vscnt", "dscnt", "kmcnt", "wait_all"}, itemIndex);
        if (optionalBool(data, "wait_all", false, itemIndex))
            inst.addModifier(SWaitCntData(0, 0, -1, 0, 0));
        else
            inst.addModifier(SWaitCntData(optionalInteger(data, "vlcnt", -1, itemIndex),
                                          optionalInteger(data, "vscnt", -1, itemIndex), -1,
                                          optionalInteger(data, "dscnt", -1, itemIndex),
                                          optionalInteger(data, "kmcnt", -1, itemIndex)));
    }

    if (nb::handle value = find(modifiers, "swaittensorcnt"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "s_wait_tensorcnt")
            fail(itemIndex,
                 "modifier 'swaittensorcnt' is only valid on s_wait_tensorcnt");
        if (!PyDict_Check(value.ptr()))
            fail(itemIndex, "modifier 'swaittensorcnt' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data, {"value"}, itemIndex);
        int64_t count = optionalInteger(data, "value", -1, itemIndex);
        if (count < -1 || count > 127)
            fail(itemIndex, "S_WAIT_TENSORCNT value must be in [-1, 127]");
        inst.addModifier(SWaitTensorCntData(static_cast<int8_t>(count)));
    }

    if (nb::handle value = find(modifiers, "waitalu"); value.is_valid()) {
        if (std::string_view(inst.getHwInstDesc()->mnemonic) != "s_wait_alu")
            fail(itemIndex, "modifier 'waitalu' is only valid on s_wait_alu");
        if (!PyDict_Check(value.ptr())) fail(itemIndex, "modifier 'waitalu' must be a dict");
        nb::dict data = nb::borrow<nb::dict>(value);
        rejectUnknownKeys(data,
                          {"va_vdst", "va_sdst", "va_ssrc", "hold_cnt", "vm_vsrc",
                           "va_vcc", "sa_sdst"},
                          itemIndex);
        int64_t vaVdst = optionalInteger(data, "va_vdst", -1, itemIndex);
        int64_t vaSdst = optionalInteger(data, "va_sdst", -1, itemIndex);
        int64_t vaSsrc = optionalInteger(data, "va_ssrc", -1, itemIndex);
        int64_t holdCnt = optionalInteger(data, "hold_cnt", -1, itemIndex);
        int64_t vmVsrc = optionalInteger(data, "vm_vsrc", -1, itemIndex);
        int64_t vaVcc = optionalInteger(data, "va_vcc", -1, itemIndex);
        int64_t saSdst = optionalInteger(data, "sa_sdst", -1, itemIndex);
        auto inRange = [](int64_t value, int64_t maximum) {
            return value == -1 || (value >= 0 && value <= maximum);
        };
        if (!inRange(vaVdst, 15) || !inRange(vaSdst, 7) || !inRange(vmVsrc, 7) ||
            !inRange(vaSsrc, 1) || !inRange(holdCnt, 1) || !inRange(vaVcc, 1) ||
            !inRange(saSdst, 1))
            fail(itemIndex, "S_WAIT_ALU modifier field is outside its hardware range");
        inst.addModifier(SWaitAluData(vaVdst, vaSdst, vaSsrc, holdCnt, vmVsrc, vaVcc, saSdst));
    }
}

bool hasLabelField(const HwInstDesc& desc) {
    return std::any_of(desc.operandFields.begin(), desc.operandFields.end(),
                       [](const HwInstDesc::OperandFieldDesc& field) {
                           return field.fieldType == FieldType::label;
                       });
}

bool selectPromotedFields(const nb::dict& record, const HwInstDesc& desc,
                          const std::string& mnemonic, size_t itemIndex) {
    const std::string form = requireString(record, "form", itemIndex);
    const std::string primary = "gfx1250::" + mnemonic;
    if (form == primary) return false;
    if (form == primary + "::vop3") {
        if (desc.promotedFormat == MicrocodeFormat::NONE || desc.promotedFields.empty())
            fail(itemIndex, "form '" + form + "' requests a promoted encoding that instruction '" +
                                mnemonic + "' does not provide");
        return true;
    }

    const bool selectedExpansion =
        form == primary + "::terminator" ||
        (mnemonic == "s_nop" && form == primary + "::mode_hazard") ||
        (mnemonic == "s_set_vgpr_msb" && form == primary + "::generated") ||
        (mnemonic == "v_nop" && form == primary + "::from_repeat") ||
        ((mnemonic == "s_wait_dscnt" || mnemonic == "s_wait_kmcnt" ||
          mnemonic == "s_wait_loadcnt" || mnemonic == "s_wait_storecnt") &&
         form == primary + "::from_s_waitcnt");
    if (selectedExpansion) return false;

    fail(itemIndex, "form '" + form + "' does not select instruction '" + mnemonic + "'");
}

void validateOperandCounts(const HwInstDesc& desc, size_t destCount, size_t srcCount,
                           bool promoted, size_t itemIndex) {
    size_t expectedDest = 0;
    size_t expectedSrc = 0;
    const auto fields = promoted ? desc.promotedFields : desc.operandFields;
    for (const auto& field : fields) {
        if (field.isDest)
            ++expectedDest;
        else
            ++expectedSrc;
    }
    const bool tensorSpec = (std::string_view(desc.mnemonic) == "tensor_load_to_lds" ||
                             std::string_view(desc.mnemonic) == "tensor_store_from_lds") &&
                            destCount == 0 && srcCount == 2;
    if (!tensorSpec && (destCount != expectedDest || srcCount != expectedSrc)) {
        fail(itemIndex, "instruction '" + std::string(desc.mnemonic) + "' expects " +
                            std::to_string(expectedDest) + " dst and " +
                            std::to_string(expectedSrc) + " src operands, got " +
                            std::to_string(destCount) + " dst and " +
                            std::to_string(srcCount) + " src");
    }
}

bool isScalarRegister(RegType type) {
    return type == RegType::S || type == RegType::SCC || type == RegType::VCC ||
           type == RegType::VCC_LO || type == RegType::VCC_HI || type == RegType::EXEC ||
           type == RegType::EXEC_LO || type == RegType::EXEC_HI;
}

bool isM0(const StinkyRegister& operand) {
    return operand.dataType == StinkyRegister::Type::Register && operand.reg.type == RegType::M &&
           operand.reg.idx == 0 && operand.reg.num == 1;
}

bool fieldRequiresVgpr(FieldType type) {
    return type == FieldType::vgpr || type == FieldType::src_vgpr;
}

bool fieldRequiresScalar(FieldType type) {
    return type == FieldType::sreg || type == FieldType::sreg_m0 ||
           type == FieldType::sgpr || type == FieldType::sdst || type == FieldType::ssrc;
}

bool fieldRequiresPhysicalScalar(FieldType type) {
    return type == FieldType::sreg || type == FieldType::sreg_m0 ||
           type == FieldType::sgpr || type == FieldType::sdst;
}

bool fieldRequiresInteger(FieldType type) {
    return type == FieldType::simm16 || type == FieldType::simm32 ||
           type == FieldType::simm24 || type == FieldType::simm5 ||
           type == FieldType::set_vgpr_msb || type == FieldType::sleep ||
           type == FieldType::ssrc_barrier_id || type == FieldType::delay ||
           type == FieldType::smem_offset_nok ||
           type == FieldType::wait_alu || type == FieldType::wait_mem_ds;
}

bool isNamedScalarSpecial(const std::string& value) {
    constexpr std::string_view prefix = "ttmp";
    if (!value.starts_with(prefix) || value.size() == prefix.size()) return false;
    if (!std::all_of(value.begin() + static_cast<std::ptrdiff_t>(prefix.size()), value.end(),
                     [](char c) { return std::isdigit(static_cast<unsigned char>(c)); }))
        return false;
    unsigned index = 0;
    for (auto it = value.begin() + static_cast<std::ptrdiff_t>(prefix.size()); it != value.end(); ++it)
        index = index * 10 + static_cast<unsigned>(*it - '0');
    return index < 16;
}

bool allowsTtmp(const HwInstDesc& desc, bool isDest, size_t operandIndex,
                FieldType fieldType) {
    if (isDest || operandIndex != 0 || fieldType != FieldType::ssrc) return false;
    const std::string_view mnemonic(desc.mnemonic);
    return mnemonic == "s_and_b32" || mnemonic == "s_bfe_u32" ||
           mnemonic == "s_mov_b32";
}

bool allowsM0(const HwInstDesc& desc, bool isDest, size_t operandIndex,
              FieldType fieldType) {
    const std::string_view mnemonic(desc.mnemonic);
    if (mnemonic == "s_mov_b32")
        return isDest && operandIndex == 0 && fieldType == FieldType::sdst;
    if (mnemonic == "s_prefetch_inst")
        return !isDest && operandIndex == 2 && fieldType == FieldType::smem_offset_nok;
    return fieldType == FieldType::sreg_m0;
}

void validateRegisterRange(const StinkyRegister& operand, const ArchHelper::ArchInfo& archInfo,
                           const std::string& location, size_t itemIndex,
                           bool allowAuditedVgprBoundaryCrossing = false) {
    if (operand.reg.num == 0) fail(itemIndex, location + " register count must be positive");
    if (operand.reg.offset != 0)
        fail(itemIndex, location +
                            " must already be low-bit materialized; Register.offset must be zero");

    // A VGPR operand field encodes 8 bits; registers past v255 are reached through
    // s_set_vgpr_msb, which the caller plans. archInfo.maxVGPR counts the whole
    // MSB-addressable file (1024 on gfx1250), so it cannot bound an encoded operand.
    constexpr uint64_t kDirectlyEncodedVgprs = 256;
    const uint64_t end = static_cast<uint64_t>(operand.reg.idx) + operand.reg.num;
    if (operand.reg.type == RegType::V &&
        end > std::min<uint64_t>(archInfo.maxVGPR, kDirectlyEncodedVgprs) &&
        !allowAuditedVgprBoundaryCrossing)
        fail(itemIndex, location + " VGPR range exceeds the directly encoded target range");
    if (operand.reg.type == RegType::S && end > archInfo.maxSGPR)
        fail(itemIndex, location + " SGPR range exceeds the target range");
    if ((operand.reg.type == RegType::A || operand.reg.type == RegType::ACC ||
         operand.reg.type == RegType::AGPR) &&
        (archInfo.maxAGPR == 0 || end > archInfo.maxAGPR))
        fail(itemIndex, location + " AGPR range is unsupported on this target");
}

void validateOperand(const StinkyRegister& operand, const HwInstDesc& desc,
                     const HwInstDesc::OperandFieldDesc& field,
                     const ArchHelper::ArchInfo& archInfo, GfxArchID arch, bool isDest,
                     size_t operandIndex, bool vglobalSaddrOff, const std::string& location,
                     size_t itemIndex) {
    if (operand.dataType == StinkyRegister::Type::Invalid)
        fail(itemIndex, location + " cannot be an invalid/null Register");
    if (operand.dataType == StinkyRegister::Type::HwReg) {
        if (field.fieldType != FieldType::hwreg)
            fail(itemIndex, location + " hwreg operand is only valid for a hardware-register field");
        if (operand.hwreg.size == 0 || operand.hwreg.offset + operand.hwreg.size > 32)
            fail(itemIndex, location + " hwreg slice must fit in 32 bits");
        if (HwReg::idToName(arch, operand.hwreg.id).empty())
            fail(itemIndex, location + " hwreg id is not defined for the selected target");
        return;
    }
    if (field.fieldType == FieldType::hwreg)
        fail(itemIndex, location + " hardware-register field requires a structured hwreg operand");
    if (field.isDest && operand.dataType != StinkyRegister::Type::Register)
        fail(itemIndex, location + " destination must be a physical register");

    if (operand.dataType == StinkyRegister::Type::LiteralString) {
        if (field.fieldType == FieldType::label) {
            if (!isLabelName(operand.literalValue))
                fail(itemIndex, location + " is not a valid label name");
            return;
        }
        const std::string_view mnemonic(desc.mnemonic ? desc.mnemonic : "");
        if (operand.literalValue == "off" && field.encodeField == EncodeField::saddr &&
            (desc.microcode == MicrocodeFormat::MC_VGLOBAL || mnemonic == "flat_load_b32" ||
             mnemonic == "flat_store_b32"))
            return;
        if (!isNamedScalarSpecial(operand.literalValue) ||
            !allowsTtmp(desc, isDest, operandIndex, field.fieldType))
            fail(itemIndex, location +
                                " string literals are only allowed for an audited TTMP scalar "
                                "source field or a compatible typed label");
        return;
    }

    if (field.fieldType == FieldType::label)
        fail(itemIndex, location + " label field requires a string target");

    if (isM0(operand)) {
        if (!allowsM0(desc, isDest, operandIndex, field.fieldType))
            fail(itemIndex, location + " m0 is not valid for this operand field");
        return;
    }

    if (field.fieldType == FieldType::smem_offset) {
        if (operand.dataType == StinkyRegister::Type::LiteralInt) return;
        if (operand.dataType != StinkyRegister::Type::Register ||
            operand.reg.type != RegType::S)
            fail(itemIndex, location + " SMEM offset requires an integer literal or SGPR");
        validateRegisterRange(operand, archInfo, location, itemIndex);
        if (operand.reg.num != 1)
            fail(itemIndex, location + " SMEM offset SGPR must have register width 1");
        return;
    }

    // s_prefetch_inst's length field (smem_offset_nok) is documented as "SGPR, M0, or
    // null", not a pure integer; m0 is already handled above via allowsM0, so only a
    // plain 1-register-wide SGPR needs an extra carve-out here before falling into the
    // integer-only check below.
    if (field.fieldType == FieldType::smem_offset_nok &&
        operand.dataType == StinkyRegister::Type::Register && operand.reg.type == RegType::S) {
        validateRegisterRange(operand, archInfo, location, itemIndex);
        if (operand.reg.num != 1)
            fail(itemIndex, location + " SMEM length SGPR must have register width 1");
        return;
    }

    if (fieldRequiresInteger(field.fieldType)) {
        if (operand.dataType != StinkyRegister::Type::LiteralInt)
            fail(itemIndex, location + " immediate field requires an integer Register literal");
        const int64_t value = operand.literalInt;
        if (field.fieldType == FieldType::set_vgpr_msb && (value < 0 || value > 0xffff))
            fail(itemIndex, location + " s_set_vgpr_msb immediate must be in [0, 65535]");
        if (field.fieldType == FieldType::simm16 && (value < -32768 || value > 65535))
            fail(itemIndex, location + " 16-bit immediate is out of range");
        if (field.fieldType == FieldType::simm24 &&
            (value < -(1 << 23) || value > (1 << 24) - 1))
            fail(itemIndex, location + " 24-bit signed/raw immediate is out of range");
        if (field.fieldType == FieldType::simm5 && (value < 0 || value > 31))
            fail(itemIndex, location + " 5-bit immediate is out of range");
        return;
    }

    if (operand.dataType != StinkyRegister::Type::Register) {
        if (fieldRequiresVgpr(field.fieldType) || fieldRequiresPhysicalScalar(field.fieldType) ||
            field.fieldType == FieldType::vcc || field.fieldType == FieldType::exec)
            fail(itemIndex, location + " field requires a physical register");
        return;
    }

    const bool auditedDsLoadB128BoundaryCrossing =
        std::string_view(desc.mnemonic) == "ds_load_b128" && isDest && operandIndex == 0 &&
        operand.reg.type == RegType::V && operand.reg.idx == 254 && operand.reg.num == 4;
    validateRegisterRange(operand, archInfo, location, itemIndex,
                          auditedDsLoadB128BoundaryCrossing);
    const bool auditedNegatedVfmaSrc0 =
        std::string_view(desc.mnemonic) == "v_fma_f32" && !isDest && operandIndex == 0 &&
        operand.reg.type == RegType::V && operand.reg.isMinus && !operand.reg.isAbs;
    if ((operand.reg.isMinus || operand.reg.isAbs) && !auditedNegatedVfmaSrc0)
        fail(itemIndex, location +
                            " per-operand negate/absolute flags are unsupported outside the "
                            "audited v_fma_f32 VGPR src0 form");
    if (fieldRequiresVgpr(field.fieldType) && operand.reg.type != RegType::V)
        fail(itemIndex, location + " field requires a VGPR");
    if (field.fieldType == FieldType::sreg_m0 && !isM0(operand))
        fail(itemIndex, location + " field requires m0");
    if (fieldRequiresScalar(field.fieldType) && field.fieldType != FieldType::sreg_m0 &&
        !isScalarRegister(operand.reg.type))
        fail(itemIndex, location + " field requires a scalar register");
    if (desc.microcode == MicrocodeFormat::MC_VOPC && isDest && operandIndex == 0 &&
        operand.reg.type != RegType::S && operand.reg.type != RegType::VCC &&
        operand.reg.type != RegType::VCC_LO && operand.reg.type != RegType::VCC_HI)
        fail(itemIndex, location + " VOPC destination requires an SGPR or VCC register");
    if (field.fieldType == FieldType::vcc && operand.reg.type != RegType::VCC &&
        operand.reg.type != RegType::VCC_LO && operand.reg.type != RegType::VCC_HI)
        fail(itemIndex, location + " field requires VCC");
    if (field.fieldType == FieldType::exec && operand.reg.type != RegType::EXEC &&
        operand.reg.type != RegType::EXEC_LO && operand.reg.type != RegType::EXEC_HI)
        fail(itemIndex, location + " field requires EXEC");

    if (field.fieldSizeBits != 0) {
        uint16_t expectedCount = std::max<uint16_t>(1, (field.fieldSizeBits + 31) / 32);
        if (vglobalSaddrOff && !isDest && field.encodeField == EncodeField::vaddr)
            expectedCount = 2;
        const bool wave32M64 = field.isM64 && archInfo.waveFrontSize == 32 &&
                               expectedCount == 2 && operand.reg.num == 1;
        if (operand.reg.num != expectedCount && !wave32M64)
            fail(itemIndex, location + " has register width " + std::to_string(operand.reg.num) +
                                ", expected " + std::to_string(expectedCount));
    }
}

void validateOperands(const HwInstDesc& desc, const std::vector<StinkyRegister>& dests,
                      const std::vector<StinkyRegister>& srcs,
                      const ArchHelper::ArchInfo& archInfo, GfxArchID arch,
                      bool promoted, size_t itemIndex) {
    size_t destIndex = 0;
    size_t srcIndex = 0;
    const auto fields = promoted ? desc.promotedFields : desc.operandFields;
    const bool vglobalSaddrOff = hasVGlobalSaddrOff(desc, srcs);
    for (const auto& field : fields) {
        const bool isDest = field.isDest;
        const size_t operandIndex = isDest ? destIndex++ : srcIndex++;
        const auto& operands = isDest ? dests : srcs;
        if (operandIndex >= operands.size()) {
            if ((std::string_view(desc.mnemonic) == "tensor_load_to_lds" ||
                 std::string_view(desc.mnemonic) == "tensor_store_from_lds") &&
                !isDest &&
                srcs.size() == 2)
                continue;
            fail(itemIndex, "internal operand-count validation mismatch");
        }
        validateOperand(operands.at(operandIndex), desc, field, archInfo, arch,
                        isDest, operandIndex, vglobalSaddrOff,
                        std::string(isDest ? "dst[" : "src[") +
                            std::to_string(operandIndex) + "]",
                        itemIndex);
    }
}

std::string trim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t");
    return value.substr(first, last - first + 1);
}

// Like trim(), but also strips \r\n: used on the single-instruction canonical text
// generated for the "encoding" raw-word override path, which must collapse to exactly
// one line before it is embedded in a trailing `// ...` comment.
std::string trimWhitespace(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool isHexDigit(char c) {
    return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

std::string validateCustomEncoding(std::string text, size_t itemIndex) {
    if (text.empty()) fail(itemIndex, "custom encoding text must not be empty");
    if (text.find('\0') != std::string::npos)
        fail(itemIndex, "custom encoding text must not contain NUL bytes");

    std::istringstream stream(text);
    std::string line;
    size_t lineNumber = 0;
    size_t encodedWords = 0;
    while (std::getline(stream, line)) {
        ++lineNumber;
        std::string stripped = trim(line);
        if (stripped.empty()) continue;
        constexpr std::string_view prefix = ".long 0x";
        if (!stripped.starts_with(prefix) || stripped.size() != prefix.size() + 8 ||
            !std::all_of(stripped.begin() + static_cast<std::ptrdiff_t>(prefix.size()),
                         stripped.end(), isHexDigit)) {
            fail(itemIndex, "custom line " + std::to_string(lineNumber) +
                                " must be exactly '.long 0x' followed by 8 hexadecimal digits");
        }
        ++encodedWords;
    }
    if (encodedWords == 0) fail(itemIndex, "custom encoding must contain at least one .long word");
    if (text.back() != '\n') text.push_back('\n');
    return text;
}

std::string validateStandaloneComment(std::string text, size_t itemIndex) {
    if (text.find_first_of("\r\n") != std::string::npos)
        fail(itemIndex, "field 'text' must be a single line");
    if (text.find('\0') != std::string::npos)
        fail(itemIndex, "field 'text' must not contain NUL bytes");
    return "// " + text + "\n";
}

// --- Raw-word ("encoding") override mechanism -------------------------------------
//
// StinkyTofu normally prints typed instructions as LLVM-syntax assembly text and lets
// LLVM's own assembler pick the final bits. A handful of gfx1250 instructions need
// bits that the LLVM *text* syntax cannot express: LLVM always encodes
// V_WMMA_SCALE's "Constant" field as 0x080 although the ISA requires 0x100, and some
// SP3-authored test vectors put non-zero raw values into fields the ISA defines as
// "unused" (V_NOP's SRC0; V_PK_*_F32's SRC2 when fewer than 3 sources are read).
//
// For these cases the bridge: (1) builds and validates the instruction exactly as for
// the normal typed path, (2) assembles its *canonical* (LLVM-correct) text via comgr
// to get the real bit pattern, (3) patches only the specific bits the caller named,
// after checking each one is ISA-documented as unused/constant or (for op_sel_hi)
// unread by the affected source, and (4) emits the patched words as `.long` directives
// with a comment carrying the canonical text and the overridden fields, instead of
// assembly text for this item.
//
// Bit positions are taken directly from the CDNA5 ISA document: Table 86
// (VOP1 SRC0 = bits [8:0]), Table 95 (VOP3P SRC2 = bits [58:50], OPSEL_HI[1:0] = bits
// [60:59], OPSEL_HI2 = bit [14]), and Table 97 (VOP3PX2 "Constant" = bits [58:50] -- the
// exact same bit range as VOP3P's SRC2, since VOP3PX2's 64-bit "load scale" portion
// reuses the VOP3P SRC0/SRC1/SRC2 slot layout and repurposes the SRC2 slot for this
// field). Plain VOP3 (not VOP3P) instructions share that same SRC2 bit range: verified
// with llvm-mc that v_max_u32_e64/v_min_u32_e64/v_xor_b32_e64/v_cmp_gt_i32_e64/
// v_cmp_ge_f32_e64 all place their (unread, 2-source) SRC2 field at word1 bits
// [26:18], i.e. the same global bits [58:50] as kVop3pSrc2Loc below.

struct EncodingFieldLoc {
    int bitOffset;  // bit index of the LSB, counted from bit 0 of the first 32-bit word
    int bitWidth;
};

constexpr EncodingFieldLoc kVop1Src0Loc{0, 9};
constexpr EncodingFieldLoc kVop3pSrc2Loc{50, 9};
constexpr EncodingFieldLoc kVop3pScaleConstantLoc{50, 9};
// OPSEL_HI bit for source i: src0 -> bit 59, src1 -> bit 60, src2 (OPSEL_HI2) -> bit 14.
constexpr int kVop3pOpSelHiBit[3] = {59, 60, 14};

void patchBits(std::vector<uint32_t>& words, int bitOffset, int bitWidth, uint64_t value,
              size_t itemIndex, const std::string& fieldName) {
    if (bitWidth < 64 && value >= (uint64_t{1} << bitWidth))
        fail(itemIndex, "encoding field '" + fieldName + "' does not fit in " +
                            std::to_string(bitWidth) + " bits");
    for (int b = 0; b < bitWidth; ++b) {
        const int globalBit = bitOffset + b;
        const size_t wordIdx = static_cast<size_t>(globalBit / 32);
        const int bitInWord = globalBit % 32;
        if (wordIdx >= words.size())
            fail(itemIndex, "encoding field '" + fieldName + "' bit " +
                                std::to_string(globalBit) +
                                " exceeds the instruction's encoded word count");
        const uint32_t bitValue = static_cast<uint32_t>((value >> b) & 1u);
        words[wordIdx] =
            (words[wordIdx] & ~(uint32_t{1} << bitInWord)) | (bitValue << bitInWord);
    }
}

uint32_t readBit(const std::vector<uint32_t>& words, int globalBit, size_t itemIndex) {
    const size_t wordIdx = static_cast<size_t>(globalBit / 32);
    if (wordIdx >= words.size())
        fail(itemIndex, "encoding field bit " + std::to_string(globalBit) +
                            " exceeds the instruction's encoded word count");
    return (words[wordIdx] >> (globalBit % 32)) & 1u;
}

// Applies the "encoding" dict's requested raw-bit overrides to `words` (the
// canonical, comgr-assembled encoding), with strict validation: every field may only
// be written where the ISA defines it as unused/constant for this instruction shape,
// or (op_sel_hi) where the differing bit targets a source that is absent or not a
// VGPR (so the LLVM-forced value and the caller's requested value are both
// behaviorally equivalent -- the hardware never reads that bit as a VGPR select).
// Appends a short "field=value" summary of what was overridden to `summaryOut`.
void applyEncodingOverride(std::vector<uint32_t>& words, const nb::dict& encoding,
                           const HwInstDesc& desc, const std::vector<StinkyRegister>& srcs,
                           size_t itemIndex, std::string& summaryOut) {
    rejectUnknownKeys(encoding, {"src0", "src2", "op_sel_hi", "scale_constant"}, itemIndex);
    const std::string_view mnemonic(desc.mnemonic ? desc.mnemonic : "");
    const MicrocodeFormat mc = desc.microcode;
    std::ostringstream summary;

    if (nb::handle v = find(encoding, "src0"); v.is_valid()) {
        if (mc != MicrocodeFormat::MC_VOP1)
            fail(itemIndex, "encoding field 'src0' is only valid on VOP1 instructions");
        if (!srcs.empty())
            fail(itemIndex,
                 "encoding field 'src0' is only valid when the instruction has no src "
                 "operands (SRC0 must be unread by the ISA), e.g. v_nop");
        const int64_t value = requireInteger(v, "encoding.src0", itemIndex);
        if (value < 0 || value > 0x1FF)
            fail(itemIndex, "encoding field 'src0' must be in [0, 511]");
        patchBits(words, kVop1Src0Loc.bitOffset, kVop1Src0Loc.bitWidth,
                 static_cast<uint64_t>(value), itemIndex, "src0");
        summary << " src0=0x" << std::hex << value << std::dec;
    }

    if (nb::handle v = find(encoding, "src2"); v.is_valid()) {
        if (mc != MicrocodeFormat::MC_VOP3P && mc != MicrocodeFormat::MC_VOP3)
            fail(itemIndex,
                 "encoding field 'src2' is only valid on VOP3 or VOP3P instructions");
        if (srcs.size() >= 3)
            fail(itemIndex,
                 "encoding field 'src2' is only valid when the instruction has fewer "
                 "than 3 src operands (SRC2 must be unread by the ISA)");
        const int64_t value = requireInteger(v, "encoding.src2", itemIndex);
        if (value < 0 || value > 0x1FF)
            fail(itemIndex, "encoding field 'src2' must be in [0, 511]");
        patchBits(words, kVop3pSrc2Loc.bitOffset, kVop3pSrc2Loc.bitWidth,
                 static_cast<uint64_t>(value), itemIndex, "src2");
        summary << " src2=0x" << std::hex << value << std::dec;
    }

    if (nb::handle v = find(encoding, "scale_constant"); v.is_valid()) {
        if (!mnemonic.starts_with("v_wmma_scale"))
            fail(itemIndex,
                 "encoding field 'scale_constant' is only valid on v_wmma_scale_* "
                 "instructions");
        const int64_t value = requireInteger(v, "encoding.scale_constant", itemIndex);
        if (value < 0 || value > 0x1FF)
            fail(itemIndex, "encoding field 'scale_constant' must be in [0, 511]");
        patchBits(words, kVop3pScaleConstantLoc.bitOffset, kVop3pScaleConstantLoc.bitWidth,
                 static_cast<uint64_t>(value), itemIndex, "scale_constant");
        summary << " scale_constant=0x" << std::hex << value << std::dec;
    }

    if (nb::handle v = find(encoding, "op_sel_hi"); v.is_valid()) {
        if (mc != MicrocodeFormat::MC_VOP3P && mc != MicrocodeFormat::MC_VOP3PX2 &&
            mc != MicrocodeFormat::MC_VOP3PX3)
            fail(itemIndex,
                 "encoding field 'op_sel_hi' is only valid on VOP3P-family instructions");
        if (!PyList_Check(v.ptr()) || PyList_Size(v.ptr()) != 3)
            fail(itemIndex, "encoding field 'op_sel_hi' must be a 3-element list of 0/1");
        summary << " op_sel_hi=[";
        for (int i = 0; i < 3; ++i) {
            const int64_t bit =
                requireInteger(nb::handle(PyList_GetItem(v.ptr(), i)), "encoding.op_sel_hi", itemIndex);
            if (bit != 0 && bit != 1)
                fail(itemIndex, "encoding field 'op_sel_hi' entries must be 0 or 1");
            const uint32_t canonical = readBit(words, kVop3pOpSelHiBit[i], itemIndex);
            if (static_cast<uint32_t>(bit) != canonical) {
                const bool sourceExists = static_cast<size_t>(i) < srcs.size();
                const bool sourceIsVgpr =
                    sourceExists && srcs[static_cast<size_t>(i)].dataType == StinkyRegister::Type::Register &&
                    srcs[static_cast<size_t>(i)].reg.type == RegType::V;
                if (sourceIsVgpr)
                    fail(itemIndex,
                         "encoding field 'op_sel_hi' bit " + std::to_string(i) +
                             " differs from the canonical (LLVM-assembled) encoding on a "
                             "VGPR source; overriding it would change which half of that "
                             "VGPR is read");
            }
            patchBits(words, kVop3pOpSelHiBit[i], 1, static_cast<uint64_t>(bit), itemIndex,
                     "op_sel_hi[" + std::to_string(i) + "]");
            summary << bit << (i < 2 ? "," : "");
        }
        summary << "]";
    }

    summaryOut = summary.str();
}

// One "encoding" item awaiting its canonical words: the directive is filled in after
// every such item of the call has been assembled together.
struct PendingEncoding {
    size_t itemIndex;
    AsmDirective* directive;
    std::string canonicalText;
    size_t expectedWords;
    nb::dict encoding;
    const HwInstDesc* desc;
    std::vector<StinkyRegister> srcs;
};

std::vector<uint32_t> bytesToWords(const std::vector<uint8_t>& bytes) {
    std::vector<uint32_t> words(bytes.size() / 4);
    for (size_t w = 0; w < words.size(); ++w) {
        words[w] = static_cast<uint32_t>(bytes[w * 4]) |
                   (static_cast<uint32_t>(bytes[w * 4 + 1]) << 8) |
                   (static_cast<uint32_t>(bytes[w * 4 + 2]) << 16) |
                   (static_cast<uint32_t>(bytes[w * 4 + 3]) << 24);
    }
    return words;
}

// Fills every pending directive with its canonical words plus the requested field
// overrides. Each comgr invocation costs milliseconds, so all texts not yet seen are
// assembled in one batch and memoized across calls; the canonical words depend only on
// the ISA, the wave size and the text. A batch whose total size disagrees with the
// expected word counts is re-assembled text by text to name the offending item.
void resolvePendingEncodings(std::vector<PendingEncoding>& pending, const std::string& isaName,
                             uint32_t waveSize) {
    static std::mutex cacheMutex;
    static std::unordered_map<std::string, std::vector<uint32_t>> cache;
    const std::string keyPrefix = isaName + "\n" + std::to_string(waveSize) + "\n";
    std::lock_guard<std::mutex> lock(cacheMutex);

    std::vector<const PendingEncoding*> misses;
    std::unordered_set<std::string> queued;
    for (const PendingEncoding& entry : pending) {
        const std::string key = keyPrefix + entry.canonicalText;
        if (!cache.contains(key) && queued.insert(entry.canonicalText).second)
            misses.push_back(&entry);
    }

    if (!misses.empty()) {
        std::string batch;
        size_t expectedBytes = 0;
        for (const PendingEncoding* entry : misses) {
            batch += entry->canonicalText;
            batch += "\n";
            expectedBytes += entry->expectedWords * 4;
        }
        std::vector<uint8_t> bytes;
        if (tryAssembleToBytes(batch, isaName, waveSize, bytes) && bytes.size() == expectedBytes) {
            size_t offset = 0;
            for (const PendingEncoding* entry : misses) {
                const size_t size = entry->expectedWords * 4;
                const auto first = bytes.begin() + static_cast<std::ptrdiff_t>(offset);
                std::vector<uint8_t> slice(first, first + static_cast<std::ptrdiff_t>(size));
                cache.emplace(keyPrefix + entry->canonicalText, bytesToWords(slice));
                offset += size;
            }
        } else {
            for (const PendingEncoding* entry : misses) {
                std::vector<uint8_t> single;
                if (!tryAssembleToBytes(entry->canonicalText, isaName, waveSize, single))
                    fail(entry->itemIndex, "modifier 'encoding': comgr failed to assemble the "
                                           "canonical form '" + entry->canonicalText + "'");
                if (single.empty() || single.size() % 4 != 0)
                    fail(entry->itemIndex, "modifier 'encoding': comgr returned an unexpected "
                                           "byte count for '" + entry->canonicalText + "'");
                cache.emplace(keyPrefix + entry->canonicalText, bytesToWords(single));
            }
        }
    }

    for (PendingEncoding& entry : pending) {
        std::vector<uint32_t> words = cache.at(keyPrefix + entry.canonicalText);
        // A non-inline-constant literal source appends one trailing 32-bit word (the
        // ISA's "LITERAL" encoding); every field this bridge's "encoding" override can
        // patch lives within entry.expectedWords, so one extra trailing word is benign
        // -- accept it rather than reject the whole instruction.
        if (words.size() != entry.expectedWords && words.size() != entry.expectedWords + 1)
            fail(entry.itemIndex, "modifier 'encoding': assembled '" + entry.canonicalText +
                                      "' to " + std::to_string(words.size()) +
                                      " words, expected " + std::to_string(entry.expectedWords) +
                                      " (or " + std::to_string(entry.expectedWords + 1) +
                                      " with one trailing literal word)");
        std::string overrideSummary;
        applyEncodingOverride(words, entry.encoding, *entry.desc, entry.srcs, entry.itemIndex,
                              overrideSummary);
        std::ostringstream text;
        for (uint32_t w : words) {
            text << ".long 0x" << std::hex << std::setw(8) << std::setfill('0') << w << std::dec
                 << std::setfill(' ') << "\n";
        }
        text << "// raw-encoding override of '" << entry.canonicalText << "':" << overrideSummary
             << "\n";
        entry.directive->value = text.str();
    }
}

std::string emitAsm(const std::string& archName, uint32_t waveSize, const nb::list& items) {
    const auto& archHelper = ArchHelper::getInstance();
    const auto* archInfo = archHelper.getArchInfo(archName);
    if (!archInfo)
        throw nb::value_error(("emit_asm: unsupported architecture '" + archName + "'").c_str());
    if (waveSize != archInfo->waveFrontSize) {
        throw nb::value_error(("emit_asm: wave_size " + std::to_string(waveSize) +
                               " does not match " + archName + " wave size " +
                               std::to_string(archInfo->waveFrontSize))
                                  .c_str());
    }
    const GfxArchID arch = archHelper.getGfxArchID(archName);
    static constexpr char kHexDigits[] = "0123456789abcdef";
    std::string isaName = "amdgcn-amd-amdhsa--gfx";
    isaName += std::to_string(archInfo->major);
    isaName += std::to_string(archInfo->minor);
    isaName += kHexDigits[archInfo->stepping & 0xF];
    std::vector<PendingEncoding> pendingEncodings;

    // Promoted descriptors and their mnemonic storage must outlive every IR
    // object that refers to them, including Function destruction.
    std::vector<std::unique_ptr<std::string>> selectedMnemonicStorage;
    std::vector<std::unique_ptr<HwInstDesc>> selectedDescriptorStorage;
    Function function("emit_asm");
    BasicBlock* block = function.createBasicBlock("entry");
    AsmIRBuilder builder(*block, arch);
    size_t typedInstructionCount = 0;
    bool requiresM0WidthBypass = false;
    std::unordered_set<std::string> labels;
    std::vector<std::pair<std::string, size_t>> branchTargets;

    size_t itemIndex = 0;
    for (nb::handle item : items) {
        if (!PyDict_Check(item.ptr())) fail(itemIndex, "record must be a dict");
        nb::dict record = nb::borrow<nb::dict>(item);
        const std::string kind = requireString(record, "kind", itemIndex);

        if (kind == "instruction") {
            rejectUnknownKeys(record, {"kind", "form", "mnemonic", "dst", "src", "target",
                                       "modifiers", "comment"}, itemIndex);
            const std::string mnemonic = requireString(record, "mnemonic", itemIndex);
            const auto& mnemonicMap = archInfo->getMnemonicToIsaOpcodeMap();
            auto opcodeIt = mnemonicMap.find(mnemonic);
            if (opcodeIt == mnemonicMap.end())
                fail(itemIndex, "unknown instruction '" + mnemonic + "' for " + archName);
            if (!isSupportedTypedInstruction(mnemonic))
                fail(itemIndex, "instruction '" + mnemonic +
                                    "' is unsupported by the T01 typed bridge allowlist");
            const HwInstDesc* desc = getMCIDByIsaOp(opcodeIt->second, arch);
            if (!desc || !desc->mnemonic)
                fail(itemIndex, "instruction '" + mnemonic + "' has no hardware descriptor");
            const bool promoted = selectPromotedFields(record, *desc, mnemonic, itemIndex);

            auto dests = parseOperands(record, "dst", itemIndex);
            auto srcs = parseOperands(record, "src", itemIndex);
            nb::handle targetValue = find(record, "target");
            if (hasLabelField(*desc)) {
                if (!targetValue.is_valid())
                    fail(itemIndex, "branch instruction '" + mnemonic + "' requires 'target'");
                if (!srcs.empty())
                    fail(itemIndex, "branch instruction '" + mnemonic +
                                        " uses 'target'; explicit src operands are not allowed");
                if (!PyUnicode_Check(targetValue.ptr()))
                    fail(itemIndex, "field 'target' must be a string");
                std::string target = nb::cast<std::string>(targetValue);
                if (!isLabelName(target)) fail(itemIndex, "field 'target' is not a valid label name");
                srcs.emplace_back(target);
                branchTargets.emplace_back(target, itemIndex);
            } else if (targetValue.is_valid()) {
                fail(itemIndex,
                     "field 'target' is only valid for an instruction with a label operand");
            }
            validateOperandCounts(*desc, dests.size(), srcs.size(), promoted, itemIndex);
            validateOperands(*desc, dests, srcs, *archInfo, arch, promoted, itemIndex);
            requiresM0WidthBypass = requiresM0WidthBypass ||
                                    std::any_of(dests.begin(), dests.end(), isM0) ||
                                    std::any_of(srcs.begin(), srcs.end(), isM0);

            const HwInstDesc* selectedDesc = desc;
            if (promoted) {
                selectedMnemonicStorage.push_back(
                    std::make_unique<std::string>(mnemonic + "_e64"));
                auto selected = std::make_unique<HwInstDesc>(*desc);
                selected->mnemonic = selectedMnemonicStorage.back()->c_str();
                selected->microcode = desc->promotedFormat;
                selected->operandFields = desc->promotedFields;
                selected->promotedFormat = MicrocodeFormat::NONE;
                selected->promotedFields = {};
                selectedDescriptorStorage.push_back(std::move(selected));
                selectedDesc = selectedDescriptorStorage.back().get();
            }

            nb::handle modifiersHandle = find(record, "modifiers");
            nb::dict modifiersDict = (modifiersHandle.is_valid() && PyDict_Check(modifiersHandle.ptr()))
                                         ? nb::borrow<nb::dict>(modifiersHandle)
                                         : nb::dict();
            nb::handle encodingHandle = find(modifiersDict, "encoding");

            if (encodingHandle.is_valid()) {
                if (!PyDict_Check(encodingHandle.ptr()))
                    fail(itemIndex, "modifier 'encoding' must be a dict");
                if (hasLabelField(*desc))
                    fail(itemIndex, "modifier 'encoding' is not valid on a branch instruction");

                // Build this single instruction (with every *other* modifier applied)
                // in a throwaway function, purely to obtain its canonical assembly
                // text; "encoding" itself is accepted but ignored by addModifiers.
                Function scratchFunction("emit_asm_encoding_scratch");
                BasicBlock* scratchBlock = scratchFunction.createBasicBlock("entry");
                AsmIRBuilder scratchBuilder(*scratchBlock, arch);
                StinkyInstruction* scratchInst = scratchBuilder.create(selectedDesc);
                scratchInst->setDestRegs(dests);
                scratchInst->setSrcRegs(srcs);
                addModifiers(*scratchInst, record, itemIndex);

                AsmEmitterOptions scratchOptions;
                scratchOptions.emitCycleInfo = false;
                scratchOptions.emitBlankLines = false;
                scratchOptions.useSymbolicNames = false;
                StinkyAsmEmitter scratchEmitter(scratchOptions);
                std::string canonicalText = trimWhitespace(scratchEmitter.emit(scratchFunction));

                size_t expectedWords = 0;
                if (selectedDesc->microcode == MicrocodeFormat::MC_VOP1)
                    expectedWords = 1;
                else if (selectedDesc->microcode == MicrocodeFormat::MC_VOP3P ||
                         selectedDesc->microcode == MicrocodeFormat::MC_VOP3)
                    expectedWords = 2;
                else if (selectedDesc->microcode == MicrocodeFormat::MC_VOP3PX2 ||
                         selectedDesc->microcode == MicrocodeFormat::MC_VOP3PX3)
                    expectedWords = 4;
                else
                    fail(itemIndex, "modifier 'encoding' is not supported for this "
                                     "instruction's microcode format");

                AsmDirective* directive = builder.createIR<AsmDirective>();
                directive->kind = AsmDirectiveKind::TEXTBLOCK;
                pendingEncodings.push_back({itemIndex, directive, std::move(canonicalText),
                                            expectedWords, nb::borrow<nb::dict>(encodingHandle),
                                            selectedDesc, srcs});
            } else {
                StinkyInstruction* inst = builder.create(selectedDesc);
                inst->setDestRegs(dests);
                inst->setSrcRegs(srcs);
                addModifiers(*inst, record, itemIndex);
                if (nb::handle comment = find(record, "comment"); comment.is_valid()) {
                    if (!PyUnicode_Check(comment.ptr()))
                        fail(itemIndex, "field 'comment' must be a string");
                    std::string text = nb::cast<std::string>(comment);
                    if (text.find_first_of("\r\n") != std::string::npos)
                        fail(itemIndex, "field 'comment' must be a single line");
                    inst->addModifier(CommentData{text});
                }
                ++typedInstructionCount;
            }
        } else if (kind == "label") {
            rejectUnknownKeys(record, {"kind", "name", "alignment"}, itemIndex);
            std::string name = requireString(record, "name", itemIndex);
            if (!isLabelName(name)) fail(itemIndex, "field 'name' is not a valid label name");
            if (!labels.insert(name).second) fail(itemIndex, "duplicate label '" + name + "'");
            int64_t alignment = 1;
            if (nb::handle value = find(record, "alignment"); value.is_valid())
                alignment = requireInteger(value, "alignment", itemIndex);
            if (alignment < 1 || alignment > std::numeric_limits<uint16_t>::max())
                fail(itemIndex, "field 'alignment' must be in [1, 65535]");
            if ((alignment & (alignment - 1)) != 0)
                fail(itemIndex, "field 'alignment' must be a power of two");
            builder.createLabel(name, static_cast<uint16_t>(alignment));
        } else if (kind == "alignment") {
            rejectUnknownKeys(record, {"kind", "bytes"}, itemIndex);
            int64_t alignment =
                requireInteger(require(record, "bytes", itemIndex), "bytes", itemIndex);
            if (alignment < 1 || alignment > std::numeric_limits<int32_t>::max())
                fail(itemIndex, "field 'bytes' must be in [1, 2147483647]");
            if ((alignment & (alignment - 1)) != 0)
                fail(itemIndex, "field 'bytes' must be a power of two");
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::ALIGN;
            directive->name = ".align";
            directive->intValue = alignment;
        } else if (kind == "comment") {
            rejectUnknownKeys(record, {"kind", "text"}, itemIndex);
            std::string text = validateStandaloneComment(
                requireString(record, "text", itemIndex), itemIndex);
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = std::move(text);
        } else if (kind == "blank") {
            rejectUnknownKeys(record, {"kind"}, itemIndex);
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = "\n";
        } else if (kind == "custom") {
            rejectUnknownKeys(record, {"kind", "text"}, itemIndex);
            std::string text =
                validateCustomEncoding(requireString(record, "text", itemIndex), itemIndex);
            AsmDirective* directive = builder.createIR<AsmDirective>();
            directive->kind = AsmDirectiveKind::TEXTBLOCK;
            directive->value = std::move(text);
        } else {
            fail(itemIndex, "unknown kind '" + kind + "'");
        }
        ++itemIndex;
    }

    if (!pendingEncodings.empty()) resolvePendingEncodings(pendingEncodings, isaName, waveSize);

    for (const auto& [target, sourceItem] : branchTargets) {
        if (!labels.contains(target)) fail(sourceItem, "undefined branch target '" + target + "'");
    }

    if (typedInstructionCount > 0) {
        AsmVerifierConfig config;
        config.abortOnError = false;
        // The upstream verifier models scalar fields as SGPR-like and does not
        // include RegType::M.  validateOperands above has already checked every
        // descriptor field's bank, range, and width; isM0 additionally requires
        // the exact one-register m0 spelling before this narrow verifier bypass.
        config.checkRegisterWidths = !requiresM0WidthBypass;
        config.checkRegisterRanges = false;
        config.checkReadWriteOperands = false;
        std::string error = validateStinkyIR(function, config);
        if (!error.empty())
            throw nb::value_error(("emit_asm: assembly IR validation failed: " + error).c_str());
    }

    AsmEmitterOptions options;
    options.emitCycleInfo = false;
    options.emitBlankLines = false;
    options.useSymbolicNames = false;
    StinkyAsmEmitter emitter(options);
    return emitter.emit(function);
}

}  // namespace

void init_asm_bridge(nb::module_& m) {
    m.def("emit_asm", &emitAsm, nb::arg("arch"), nb::arg("wave_size"), nb::arg("items"),
          R"doc(Emit ordered, already-selected assembly records through temporary Asm IR.

Instruction records use existing Register values and exact descriptor mnemonics. The bridge
does not select forms, allocate registers, insert VGPR-MSB state, or run optimization passes.
Standalone comment and blank records become safe TEXTBLOCK directives; comments are always prefixed
with `// ` after rejecting multiline/NUL payloads, while blanks accept no payload. Custom records are
restricted to one or more 32-bit `.long 0xXXXXXXXX` encoding directives. Their bits are trusted output
from an upstream audited typed encoder; this bridge validates spelling and ordering, not the encoded
instruction semantics or control flow.
)doc");
}
