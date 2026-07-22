# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Package resource helpers for TensileLite generator data files."""

from importlib import resources
from pathlib import Path
import shutil
from typing import ContextManager, List, Tuple


_PACKAGE = __package__

_STATIC_HEADER_NAMES: Tuple[str, ...] = (
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "KernelHeader.h",
    "ReductionTemplate.h",
    "memory_gfx.h",
)


def _root():
    return resources.files(_PACKAGE)


def _resource(*parts):
    """Return a package resource under the Tensile package root."""
    item = _root()
    for part in parts:
        item = item / part
    return item


def _resource_text(*parts, encoding: str = "utf-8") -> str:
    """Read a package resource under the Tensile package root as text."""
    return _resource(*parts).read_text(encoding=encoding)


def _resource_path(*parts) -> ContextManager[Path]:
    """Return a context manager yielding a filesystem path for a resource."""
    return resources.as_file(_resource(*parts))


def _source():
    return _resource("Source")


def _custom_kernels():
    return _resource("CustomKernels")


def static_header_names() -> Tuple[str, ...]:
    """Return the static header filenames copied into generated libraries."""
    return _STATIC_HEADER_NAMES


def copy_static_headers(output_dir) -> List[str]:
    """Copy bundled static headers into output_dir and return copied filenames."""
    copied = []
    for name in static_header_names():
        with _resource_path("Source", name) as source_path:
            shutil.copy(source_path, output_dir)
        copied.append(name)
    return copied


def legacy_source_tree_path() -> Path:
    """Return the unpacked ``Tensile/Source`` path for legacy path-only callers.

    This is a compatibility escape hatch while raw ``__file__`` and package
    directory accesses are being centralized. It intentionally only works when
    package resources are unpacked on disk; it is not the resource-access shape
    we want for a future proper Python package.

    ``Tensile/Source`` also appears to be legacy in current TensileLite: the
    supported client build is driven by ``invoke build-client``/CMake from the
    project root, while this directory contains static headers and scripts but
    no ``CMakeLists.txt``. New code should prefer content-oriented helpers in
    this module, such as ``copy_static_headers()``, ``custom_kernel_text()``,
    ``known_bugs_text()``, and ``ductile_defaults_text()``, or add another
    domain-specific helper here instead of consuming this path directly.
    """
    source = _source()
    if isinstance(source, Path):
        return source
    raise RuntimeError(
        "Tensile Source resources are not available as an unpacked filesystem "
        "path. legacy_source_tree_path() is only for legacy path-only callers; "
        "add or use a content-oriented helper in Tensile.Resources instead."
    )


def custom_kernel_names() -> List[str]:
    """Return bundled custom kernel names without the .s suffix."""
    return sorted(
        resource.name[:-2]
        for resource in _custom_kernels().iterdir()
        if resource.is_file() and resource.name.endswith(".s")
    )


def custom_kernel_text(name: str) -> str:
    """Read a bundled custom kernel assembly resource."""
    return _resource_text("CustomKernels", f"{name}.s")


def known_bugs_text() -> str:
    """Read the bundled TensileLogic known-bugs YAML resource."""
    return _resource_text("TensileLogic", "known_bugs.yaml")


def ductile_defaults_text() -> str:
    """Read the bundled ductile defaults YAML resource."""
    return _resource_text("ductile", "config", "defaults.yaml")
