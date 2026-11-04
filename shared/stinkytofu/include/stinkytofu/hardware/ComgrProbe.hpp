// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {

/// Try to assemble the given string against the specified ISA using comgr.
/// Returns true if assembly succeeds, false on failure or if comgr is unavailable.
STINKYTOFU_EXPORT bool tryAssembleWithComgr(const std::string& asmString,
                                            const std::string& isaName, uint32_t wavefrontSize);

/// Assemble `asmString` (one or more plain instruction lines, no directives required)
/// against `isaName`/`wavefrontSize` via comgr, and return the raw bytes of the
/// resulting object's `.text` section. Used by the typed-bridge raw-word ("encoding")
/// override mechanism to obtain the LLVM-canonical encoding of an instruction before
/// patching specific ISA-documented unused/constant bits.
///
/// Returns true and fills `outBytes` on success (outBytes.size() is always a multiple
/// of 4). Returns false (leaving outBytes unspecified) on any failure: comgr
/// unavailable, assembly failure, or an unexpected object layout (missing/empty
/// `.text`, 32-bit ELF, etc).
STINKYTOFU_EXPORT bool tryAssembleToBytes(const std::string& asmString, const std::string& isaName,
                                          uint32_t wavefrontSize, std::vector<uint8_t>& outBytes);

/// Returns true if comgr support was compiled into this build.
STINKYTOFU_EXPORT bool hasComgrSupport();

/// Returns true if the installed comgr lists `isaName` as a supported ISA.
/// Useful for distinguishing "assembler rejected the input" from
/// "the ISA itself is unknown to this toolchain version".
STINKYTOFU_EXPORT bool comgrSupportsIsa(const std::string& isaName);

}  // namespace stinkytofu
