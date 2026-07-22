# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
import yaml

from Tensile import Resources


pytestmark = pytest.mark.unit

EXPECTED_STATIC_HEADERS = (
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "KernelHeader.h",
    "ReductionTemplate.h",
    "memory_gfx.h",
)

KNOWN_CUSTOM_KERNEL = (
    "Custom_Cijk_Ailk_Bljk_F8NH_HHS_BH_Bias_AS_SAB_SAV_shortname0_gfx942"
)


def _fake_resource_tree(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "pkg"
    (root / "Source").mkdir(parents=True)
    (root / "CustomKernels").mkdir()
    (root / "TensileLogic").mkdir()
    monkeypatch.setattr(Resources, "_root", lambda: root)
    return root


def _write_static_headers(root: Path) -> None:
    for name in EXPECTED_STATIC_HEADERS:
        (root / "Source" / name).write_text(f"contents for {name}\n", encoding="utf-8")


def test_static_header_names_are_exact_and_immutable():
    names = Resources.static_header_names()

    assert names == EXPECTED_STATIC_HEADERS
    assert isinstance(names, tuple)

    with pytest.raises(AttributeError):
        names.append("unexpected.h")
    assert Resources.static_header_names() == EXPECTED_STATIC_HEADERS


def test_copy_static_headers_copies_expected_files_from_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    _write_static_headers(root)
    output = tmp_path / "out"
    output.mkdir()

    copied = Resources.copy_static_headers(output)

    assert copied == list(EXPECTED_STATIC_HEADERS)
    for name in EXPECTED_STATIC_HEADERS:
        assert (output / name).read_text(encoding="utf-8") == f"contents for {name}\n"


def test_copy_static_headers_raises_for_missing_resource(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    _write_static_headers(root)
    missing_name = EXPECTED_STATIC_HEADERS[2]
    (root / "Source" / missing_name).unlink()

    with pytest.raises(FileNotFoundError, match=missing_name):
        Resources.copy_static_headers(tmp_path / "out")


def test_legacy_source_tree_path_returns_file_backed_source_tree():
    path = Resources.legacy_source_tree_path()

    assert path.name == "Source"
    assert (path / "TensileTypes.h").is_file()


def test_custom_kernel_names_filters_suffixes_and_sorts(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    kernels = root / "CustomKernels"
    (kernels / "b.s").write_text("b", encoding="utf-8")
    (kernels / "a.s").write_text("a", encoding="utf-8")
    (kernels / "foo.bar.s").write_text("dot", encoding="utf-8")
    (kernels / ".hidden.s").write_text("hidden", encoding="utf-8")
    (kernels / "note.txt").write_text("ignore", encoding="utf-8")
    (kernels / "directory.s").mkdir()

    assert Resources.custom_kernel_names() == [".hidden", "a", "b", "foo.bar"]


def test_custom_kernel_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    (root / "CustomKernels" / "kernel.s").write_text("s_nop 0\n", encoding="utf-8")

    assert Resources.custom_kernel_text("kernel") == "s_nop 0\n"


def test_custom_kernel_text_raises_for_missing_resource(tmp_path, monkeypatch):
    _fake_resource_tree(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="missing.s"):
        Resources.custom_kernel_text("missing")


def test_known_bugs_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    text = "version: 1\nskips: []\n"
    (root / "TensileLogic" / "known_bugs.yaml").write_text(text, encoding="utf-8")

    assert Resources.known_bugs_text() == text


def test_ductile_defaults_text_uses_resource_root(tmp_path, monkeypatch):
    root = _fake_resource_tree(tmp_path, monkeypatch)
    defaults = root / "ductile" / "config"
    defaults.mkdir(parents=True)
    text = "runner:\n  name: pytest\n"
    (defaults / "defaults.yaml").write_text(text, encoding="utf-8")

    assert Resources.ductile_defaults_text() == text


def test_known_bugs_text_raises_for_missing_resource(tmp_path, monkeypatch):
    _fake_resource_tree(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="known_bugs.yaml"):
        Resources.known_bugs_text()


def test_real_static_header_resources_are_available(tmp_path):
    copied = Resources.copy_static_headers(tmp_path)

    assert copied == list(EXPECTED_STATIC_HEADERS)
    for name in EXPECTED_STATIC_HEADERS:
        assert (tmp_path / name).is_file()


def test_real_custom_kernel_resources_are_available():
    names = Resources.custom_kernel_names()

    assert names
    assert KNOWN_CUSTOM_KERNEL in names


def test_real_known_bugs_resource_is_parseable_yaml():
    data = yaml.safe_load(Resources.known_bugs_text())

    assert data["version"] == 1
    assert isinstance(data["skips"], list)
