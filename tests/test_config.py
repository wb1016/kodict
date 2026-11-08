"""Tests for config loading."""

import pytest

from kodict import config


def test_defaults_when_file_missing(tmp_path):
    cfg = config.load(tmp_path / "nope.toml")
    assert cfg.jobs.count == 16
    assert cfg.model.quantizations == ("fp8",)


def test_repo_config_loads():
    cfg = config.load("config.toml")
    assert cfg.model.name == "qwen/qwen3.6-35b-a3b"
    assert cfg.model.quantizations == ("fp8",)
    assert cfg.model.frequency_penalty == 0.0
    assert cfg.model.repetition_penalty == 1.1
    assert cfg.model.max_tokens == 8192
    assert cfg.jobs.count == 16
    assert cfg.output.locale == "ko"
    assert cfg.output.dicttool == "deps/aosp-dictionary-tools/dicttool_aosp.jar"


def test_partial_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[jobs]\ncount = 4\n", encoding="utf-8")
    cfg = config.load(path)
    assert cfg.jobs.count == 4
    assert cfg.model.name == config.ModelConfig.name


def test_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert config.api_key() == "sk-test"
    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.chdir(tmp_path)  # no .env here
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        config.api_key()


def test_api_key_falls_back_to_dotenv(monkeypatch, tmp_path):
    import os
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('# comment\nOPENROUTER_API_KEY="sk-from-dotenv"\n',
                                   encoding="utf-8")
    assert config.api_key() == "sk-from-dotenv"
    assert os.environ["OPENROUTER_API_KEY"] == "sk-from-dotenv"  # propagated


def test_kodict_sort_env_override(monkeypatch):
    monkeypatch.setenv("KODICT_SORT", "")       # empty -> default routing
    assert config.load("config.toml").model.sort == ""
    monkeypatch.setenv("KODICT_SORT", "throughput")
    assert config.load("config.toml").model.sort == "throughput"
    monkeypatch.delenv("KODICT_SORT")
    assert config.load("config.toml").model.sort == "price"


def test_decompose_defaults_on():
    assert config.load("config.toml").output.decompose is True
    assert config.OutputConfig().decompose is True
