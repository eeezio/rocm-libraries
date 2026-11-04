// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "stinkytofu/hardware/ComgrProbe.hpp"

#include "stinkytofu/Config/Config.h"

#ifdef STINKYTOFU_HAS_COMGR
#include <amd_comgr/amd_comgr.h>

#include <cstdint>
#include <string_view>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#include <delayimp.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

std::string getRocmBinDir() {
    char* rocmPath = nullptr;
    size_t len = 0;
    _dupenv_s(&rocmPath, &len, "ROCM_PATH");
    if (!rocmPath) return {};
    std::string binDir = std::string(rocmPath) + "\\bin";
    free(rocmPath);
    return binDir;
}

FARPROC WINAPI delayLoadHook(unsigned dliNotify, PDelayLoadInfo pdli) {
    if (dliNotify != dliNotePreLoadLibrary) return nullptr;

    std::string binDir = getRocmBinDir();
    if (binDir.empty()) return nullptr;

    std::string fullPath = binDir + "\\" + pdli->szDll;
    HMODULE mod =
        LoadLibraryExA(fullPath.c_str(), nullptr,
                       LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    return reinterpret_cast<FARPROC>(mod);
}

}  // namespace

extern "C" const PfnDliHook __pfnDliNotifyHook2 = delayLoadHook;

bool isRocmPathSet() {
    static int cached = -1;
    if (cached >= 0) return cached;
    std::string binDir = getRocmBinDir();
    if (binDir.empty()) {
        fprintf(stderr, "stinkytofu: ROCM_PATH not set, comgr probing disabled\n");
        cached = 0;
        return false;
    }
    cached = 1;
    return true;
}
#endif  // _WIN32

#endif  // STINKYTOFU_HAS_COMGR

namespace stinkytofu {

#ifdef STINKYTOFU_HAS_COMGR

namespace {

struct ComgrData {
    amd_comgr_data_t handle{};
    bool valid = false;

    ComgrData() = default;
    ~ComgrData() {
        if (valid) amd_comgr_release_data(handle);
    }
    ComgrData(const ComgrData&) = delete;
    ComgrData& operator=(const ComgrData&) = delete;

    bool create() {
        valid = (amd_comgr_create_data(AMD_COMGR_DATA_KIND_SOURCE, &handle) ==
                 AMD_COMGR_STATUS_SUCCESS);
        return valid;
    }
};

struct ComgrDataSet {
    amd_comgr_data_set_t handle{};
    bool valid = false;

    ComgrDataSet() = default;
    ~ComgrDataSet() {
        if (valid) amd_comgr_destroy_data_set(handle);
    }
    ComgrDataSet(const ComgrDataSet&) = delete;
    ComgrDataSet& operator=(const ComgrDataSet&) = delete;

    bool create() {
        valid = (amd_comgr_create_data_set(&handle) == AMD_COMGR_STATUS_SUCCESS);
        return valid;
    }
};

struct ComgrActionInfo {
    amd_comgr_action_info_t handle{};
    bool valid = false;

    ComgrActionInfo() = default;
    ~ComgrActionInfo() {
        if (valid) amd_comgr_destroy_action_info(handle);
    }
    ComgrActionInfo(const ComgrActionInfo&) = delete;
    ComgrActionInfo& operator=(const ComgrActionInfo&) = delete;

    bool create() {
        valid = (amd_comgr_create_action_info(&handle) == AMD_COMGR_STATUS_SUCCESS);
        return valid;
    }
};

}  // namespace

namespace {

// Minimal little-endian ELF64 section-header walk to find ".text" bytes in a
// relocatable object produced by AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE.
// Avoids pulling in a full ELF library for what is a handful of fixed-layout reads.
bool extractTextSection(const std::vector<char>& elf, std::vector<uint8_t>& outBytes) {
    auto readU16 = [&](size_t off) -> uint16_t {
        return static_cast<uint16_t>(static_cast<uint8_t>(elf[off]) |
                                     (static_cast<uint8_t>(elf[off + 1]) << 8));
    };
    auto readU32 = [&](size_t off) -> uint32_t {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(static_cast<uint8_t>(elf[off + i])) << (8 * i);
        return v;
    };
    auto readU64 = [&](size_t off) -> uint64_t {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(static_cast<uint8_t>(elf[off + i])) << (8 * i);
        return v;
    };

    constexpr size_t kEhdrSize = 64;
    if (elf.size() < kEhdrSize) return false;
    // e_ident: magic + EI_CLASS(4)=2(64-bit) + EI_DATA(5)=1(little-endian)
    if (static_cast<uint8_t>(elf[0]) != 0x7f || elf[1] != 'E' || elf[2] != 'L' || elf[3] != 'F')
        return false;
    if (static_cast<uint8_t>(elf[4]) != 2 || static_cast<uint8_t>(elf[5]) != 1) return false;

    const uint64_t shoff = readU64(0x28);
    const uint16_t shentsize = readU16(0x3a);
    const uint16_t shnum = readU16(0x3c);
    const uint16_t shstrndx = readU16(0x3e);
    if (shoff == 0 || shentsize < 64 || shnum == 0 || shstrndx >= shnum) return false;
    if (shoff + static_cast<uint64_t>(shentsize) * shnum > elf.size()) return false;

    auto sectionOff = [&](uint16_t index) { return shoff + static_cast<uint64_t>(index) * shentsize; };
    const uint64_t strtabSecOff = sectionOff(shstrndx);
    const uint64_t strtabOffset = readU64(strtabSecOff + 0x18);  // sh_offset
    const uint64_t strtabSize = readU64(strtabSecOff + 0x20);    // sh_size
    if (strtabOffset + strtabSize > elf.size()) return false;

    for (uint16_t i = 0; i < shnum; ++i) {
        const uint64_t secOff = sectionOff(i);
        const uint32_t nameOff = readU32(secOff + 0x00);  // sh_name
        if (strtabOffset + nameOff >= elf.size()) continue;
        const char* name = elf.data() + strtabOffset + nameOff;
        // Bounded strlen within the strtab.
        size_t maxLen = elf.size() - (strtabOffset + nameOff);
        size_t len = 0;
        while (len < maxLen && name[len] != '\0') ++len;
        if (len != 5 || std::string_view(name, len) != ".text") continue;

        const uint64_t dataOff = readU64(secOff + 0x18);   // sh_offset
        const uint64_t dataSize = readU64(secOff + 0x20);  // sh_size
        if (dataOff + dataSize > elf.size() || dataSize == 0 || dataSize % 4 != 0) return false;
        outBytes.assign(elf.begin() + static_cast<std::ptrdiff_t>(dataOff),
                        elf.begin() + static_cast<std::ptrdiff_t>(dataOff + dataSize));
        return true;
    }
    return false;
}

}  // namespace

bool tryAssembleToBytes(const std::string& asmString, const std::string& isaName,
                        uint32_t wavefrontSize, std::vector<uint8_t>& outBytes) {
#ifdef _WIN32
    if (!isRocmPathSet()) return false;
#endif
    ComgrData data;
    if (!data.create()) return false;
    if (!asmString.empty()) {
        if (amd_comgr_set_data(data.handle, asmString.size(), asmString.c_str()) !=
            AMD_COMGR_STATUS_SUCCESS)
            return false;
    }
    if (amd_comgr_set_data_name(data.handle, "probe.s") != AMD_COMGR_STATUS_SUCCESS) return false;

    ComgrDataSet inputSet, outputSet;
    if (!inputSet.create() || !outputSet.create()) return false;
    if (amd_comgr_data_set_add(inputSet.handle, data.handle) != AMD_COMGR_STATUS_SUCCESS)
        return false;

    ComgrActionInfo actionInfo;
    if (!actionInfo.create()) return false;
    if (amd_comgr_action_info_set_language(actionInfo.handle, AMD_COMGR_LANGUAGE_NONE) !=
        AMD_COMGR_STATUS_SUCCESS)
        return false;
    if (amd_comgr_action_info_set_isa_name(actionInfo.handle, isaName.c_str()) !=
        AMD_COMGR_STATUS_SUCCESS)
        return false;
    std::vector<const char*> options;
    if (wavefrontSize == 64) options.push_back("-mwavefrontsize64");
    if (!options.empty()) {
        if (amd_comgr_action_info_set_option_list(actionInfo.handle, options.data(),
                                                  options.size()) != AMD_COMGR_STATUS_SUCCESS)
            return false;
    }

    if (amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE, actionInfo.handle,
                            inputSet.handle, outputSet.handle) != AMD_COMGR_STATUS_SUCCESS)
        return false;

    amd_comgr_data_t relocatable{};
    if (amd_comgr_action_data_get_data(outputSet.handle, AMD_COMGR_DATA_KIND_RELOCATABLE, 0,
                                       &relocatable) != AMD_COMGR_STATUS_SUCCESS)
        return false;
    size_t size = 0;
    if (amd_comgr_get_data(relocatable, &size, nullptr) != AMD_COMGR_STATUS_SUCCESS || size == 0) {
        amd_comgr_release_data(relocatable);
        return false;
    }
    std::vector<char> elfBytes(size);
    bool ok = amd_comgr_get_data(relocatable, &size, elfBytes.data()) == AMD_COMGR_STATUS_SUCCESS;
    amd_comgr_release_data(relocatable);
    if (!ok) return false;

    return extractTextSection(elfBytes, outBytes);
}

bool tryAssembleWithComgr(const std::string& asmString, const std::string& isaName,
                          uint32_t wavefrontSize) {
#ifdef _WIN32
    if (!isRocmPathSet()) return false;
#endif
    ComgrData data;
    if (!data.create()) return false;

    if (!asmString.empty()) {
        if (amd_comgr_set_data(data.handle, asmString.size(), asmString.c_str()) !=
            AMD_COMGR_STATUS_SUCCESS)
            return false;
    }
    if (amd_comgr_set_data_name(data.handle, "probe.s") != AMD_COMGR_STATUS_SUCCESS) return false;

    ComgrDataSet inputSet, outputSet;
    if (!inputSet.create() || !outputSet.create()) return false;
    if (amd_comgr_data_set_add(inputSet.handle, data.handle) != AMD_COMGR_STATUS_SUCCESS)
        return false;

    ComgrActionInfo actionInfo;
    if (!actionInfo.create()) return false;
    if (amd_comgr_action_info_set_language(actionInfo.handle, AMD_COMGR_LANGUAGE_NONE) !=
        AMD_COMGR_STATUS_SUCCESS)
        return false;
    if (amd_comgr_action_info_set_isa_name(actionInfo.handle, isaName.c_str()) !=
        AMD_COMGR_STATUS_SUCCESS)
        return false;

    std::vector<const char*> options;
    if (wavefrontSize == 64) options.push_back("-mwavefrontsize64");
    if (!options.empty()) {
        if (amd_comgr_action_info_set_option_list(actionInfo.handle, options.data(),
                                                  options.size()) != AMD_COMGR_STATUS_SUCCESS)
            return false;
    }

    auto status = amd_comgr_do_action(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                                      actionInfo.handle, inputSet.handle, outputSet.handle);
    return status == AMD_COMGR_STATUS_SUCCESS;
}

bool hasComgrSupport() {
#ifdef _WIN32
    return isRocmPathSet();
#else
    return true;
#endif
}

bool comgrSupportsIsa(const std::string& isaName) {
#ifdef _WIN32
    if (!isRocmPathSet()) return false;
#endif
    size_t count = 0;
    if (amd_comgr_get_isa_count(&count) != AMD_COMGR_STATUS_SUCCESS) return false;
    for (size_t i = 0; i < count; ++i) {
        const char* name = nullptr;
        if (amd_comgr_get_isa_name(i, &name) != AMD_COMGR_STATUS_SUCCESS) continue;
        if (name && isaName == name) return true;
    }
    return false;
}

#else  // !STINKYTOFU_HAS_COMGR

bool tryAssembleWithComgr(const std::string& /*asmString*/, const std::string& /*isaName*/,
                          uint32_t /*wavefrontSize*/) {
    return false;
}

bool tryAssembleToBytes(const std::string& /*asmString*/, const std::string& /*isaName*/,
                        uint32_t /*wavefrontSize*/, std::vector<uint8_t>& /*outBytes*/) {
    return false;
}

bool hasComgrSupport() {
    return false;
}

bool comgrSupportsIsa(const std::string& /*isaName*/) {
    return false;
}

#endif  // STINKYTOFU_HAS_COMGR

}  // namespace stinkytofu
