"""Configuration loading: config.toml + environment."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


@dataclass(frozen=True)
class ModelConfig:
    name: str = "qwen/qwen3.6-35b-a3b"
    quantizations: tuple[str, ...] = ("fp8",)
    allow_fallbacks: bool = True
    temperature: float = 0.7
    # frequency_penalty is CUMULATIVE: it also punishes the repeated JSON
    # structure tokens ("rank", "word", braces), stopping a 400-item list
    # after ~4 items. repetition_penalty is constant per seen token instead.
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.1  # breaks word loops without killing structure
    # qwen3.6 thinks by default on some fp8 providers: the answer then lands
    # entirely in `reasoning` and `content` arrives empty — keep thinking off
    reasoning_enabled: bool = False
    # provider sorting: "price" routes to the cheapest fp8 endpoints first
    # (Prime Intellect is ~10x cheaper than DeepInfra/CoreWeave for this
    # model) — with allow_fallbacks it still falls down the list on errors
    sort: str = "price"
    max_tokens: int = 8192


@dataclass(frozen=True)
class JobsConfig:
    count: int = 16
    max_retries: int = 5


@dataclass(frozen=True)
class OutputConfig:
    locale: str = "ko"
    dict_type: str = "main"
    dict_version: int = 20
    description: str = "Korean"
    # write words NFD (canonical jamo): LeanType/HeliBoard Korean lookup
    # decomposes the typed word before the native trie search, so an NFC
    # (precomposed syllable) dictionary never matches
    decompose: bool = True
    dir: str = "out"
    dicttool: str = "deps/aosp-dictionaries/dicttool_aosp.jar"


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load config.toml; a missing file yields the defaults.

    ``KODICT_SORT`` overrides the provider sort (set it empty to fall back to
    OpenRouter's default routing, e.g. ``KODICT_SORT= kodict extract``).
    """
    path = Path(path)
    if not path.is_file():
        return Config()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    model_raw = raw.get("model", {})
    jobs_raw = raw.get("jobs", {})
    output_raw = raw.get("output", {})
    sort = os.environ.get("KODICT_SORT",
                          model_raw.get("sort", ModelConfig.sort))

    model = ModelConfig(
        name=model_raw.get("name", ModelConfig.name),
        quantizations=tuple(model_raw.get("quantizations", ModelConfig.quantizations)),
        allow_fallbacks=model_raw.get("allow_fallbacks", ModelConfig.allow_fallbacks),
        temperature=model_raw.get("temperature", ModelConfig.temperature),
        frequency_penalty=model_raw.get("frequency_penalty",
                                         ModelConfig.frequency_penalty),
        repetition_penalty=model_raw.get("repetition_penalty",
                                          ModelConfig.repetition_penalty),
        reasoning_enabled=model_raw.get("reasoning_enabled",
                                         ModelConfig.reasoning_enabled),
        sort=sort,
        max_tokens=model_raw.get("max_tokens", ModelConfig.max_tokens),
    )
    jobs = JobsConfig(
        count=jobs_raw.get("count", JobsConfig.count),
        max_retries=jobs_raw.get("max_retries", JobsConfig.max_retries),
    )
    output = OutputConfig(
        locale=output_raw.get("locale", OutputConfig.locale),
        dict_type=output_raw.get("dict_type", OutputConfig.dict_type),
        dict_version=output_raw.get("dict_version", OutputConfig.dict_version),
        description=output_raw.get("description", OutputConfig.description),
        decompose=output_raw.get("decompose", OutputConfig.decompose),
        dir=output_raw.get("dir", OutputConfig.dir),
        dicttool=output_raw.get("dicttool", OutputConfig.dicttool),
    )
    return Config(model=model, jobs=jobs, output=output)


def api_key() -> str:
    """OPENROUTER_API_KEY from the environment, falling back to ``.env``."""
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    env_path = Path(".env")
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            if name.strip() == "OPENROUTER_API_KEY":
                key = value.strip().strip("\"'")
                if key:
                    os.environ["OPENROUTER_API_KEY"] = key  # for child processes
                    return key
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set (checked the environment and .env). "
        "Get a key at https://openrouter.ai/settings/keys and put it in .env "
        "or export it before running."
    )
