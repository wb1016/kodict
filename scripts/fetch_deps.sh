#!/usr/bin/env bash
# Fetch dicttool_aosp.jar from its Apache-2.0 upstream
# (https://github.com/remi0s/aosp-dictionary-tools, AOSP LatinIME dicttool)
# using a blobless sparse clone: a few hundred KB instead of a full clone.
#
# The fetched checkout keeps its own license (Apache-2.0) and is gitignored;
# it is never distributed as part of this repository.
set -euo pipefail

URL="https://github.com/remi0s/aosp-dictionary-tools.git"
# bump this pin to pick up a newer dicttool
PIN="1e69dd2c04258e5e9e04bfb13b46faccf6a435b0"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/deps/aosp-dictionary-tools"

if [ -f "$DEST/dicttool_aosp.jar" ] && [ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null || true)" = "$PIN" ]; then
    echo "dicttool already present at $PIN"
    exit 0
fi

mkdir -p "$(dirname "$DEST")"
if [ ! -d "$DEST/.git" ]; then
    git clone --filter=blob:none --sparse --no-checkout "$URL" "$DEST"
fi

# cone mode: only root-level files are checked out (jar, LICENSE, README, ...)
git -C "$DEST" sparse-checkout set dicttool_aosp.jar
git -C "$DEST" checkout --force "$PIN"

echo "fetched aosp-dictionary-tools @ $PIN -> $DEST"
