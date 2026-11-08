"""Tests for config loading."""

from kodict import config


def test_defaults_when_file_missing(tmp_path):
    cfg = config.load(tmp_path / "nope.toml")
    assert cfg.output.locale == "ko"
    assert cfg.output.dict_type == "main"
    assert cfg.output.decompose is True


def test_repo_config_loads():
    cfg = config.load("config.toml")
    assert cfg.output.locale == "ko"
    assert cfg.output.dict_version == 20
    assert cfg.output.dicttool == "deps/aosp-dictionary-tools/dicttool_aosp.jar"


def test_partial_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[output]\nlocale = \"en\"\n", encoding="utf-8")
    cfg = config.load(path)
    assert cfg.output.locale == "en"
    assert cfg.output.dict_type == config.OutputConfig.dict_type