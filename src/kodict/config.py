"""Configuration loading: config.toml."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")


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
    dicttool: str = "deps/aosp-dictionary-tools/dicttool_aosp.jar"


@dataclass(frozen=True)
class MineConfig:
    # wikimedia/wikipedia dataset: dated subdir pinned for reproducibility
    dataset: str = "wikimedia/wikipedia"
    subdir: str = "20231101.ko"
    cache_dir: str = "state/wiki"
    out_dir: str = "state/wiki/shards"
    # unigrams: keep words seen at least this often (>=3 keeps real words,
    # drops wiki typos/one-offs), capped at the `max_words` most frequent
    min_count: int = 3
    max_words: int = 100_000
    # bigrams: only these many most-frequent eojeols may carry (parents)
    # or be suggested as (children) next words — bounds pair-count memory
    bigram_parents: int = 10_000
    bigram_children: int = 20_000
    # a parent→child link must be observed at least this many times
    min_bigram_count: int = 5
    # entries per output shard (small enough to re-read cheaply)
    shard_words: int = 10_000


@dataclass(frozen=True)
class Config:
    output: OutputConfig = field(default_factory=OutputConfig)
    mine: MineConfig = field(default_factory=MineConfig)


def load(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load config.toml; a missing file yields the defaults."""
    path = Path(path)
    if not path.is_file():
        return Config()
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    output_raw = raw.get("output", {})
    output = OutputConfig(
        locale=output_raw.get("locale", OutputConfig.locale),
        dict_type=output_raw.get("dict_type", OutputConfig.dict_type),
        dict_version=output_raw.get("dict_version", OutputConfig.dict_version),
        description=output_raw.get("description", OutputConfig.description),
        decompose=output_raw.get("decompose", OutputConfig.decompose),
        dir=output_raw.get("dir", OutputConfig.dir),
        dicttool=output_raw.get("dicttool", OutputConfig.dicttool),
    )
    mine_raw = raw.get("mine", {})
    mine = MineConfig(**{
        name: mine_raw[name]
        for name in MineConfig.__dataclass_fields__
        if name in mine_raw
    })
    return Config(output=output, mine=mine)