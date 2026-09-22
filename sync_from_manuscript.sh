#!/usr/bin/env bash
# Refresh this repository from the (private) manuscript tree.
#
# Copies ONLY the whitelisted code and exported data. It never copies the
# manuscript sources, the reference PDFs, the internal results log or the
# measured channel recordings.
set -euo pipefail
M="${1:-$HOME/Workspace/WritePaper/TWC_Akram/Manuscript_v3}"
R="$(cd "$(dirname "$0")" && pwd)"
[ -d "$M/sim" ] || { echo "manuscript tree not found: $M" >&2; exit 1; }

rsync -a --delete --exclude '__pycache__' "$M/sim/"       "$R/sim/"
rsync -a --delete                          "$M/figs/data/" "$R/figs/data/"
# DICHASUS metadata is deliberately NOT copied: third-party, licence not held.
echo "synced: $(ls "$R"/sim/*.py | wc -l) scripts, $(ls "$R"/figs/data/*.dat | wc -l) data files"
git -C "$R" status --short
