#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/yangping4271/Subtitle-Translator.git"
REPO_DIR="${SUBTITLE_TRANSLATOR_REPO_DIR:-$HOME/tools/Subtitle-Translator}"
FORCE=0

if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv 未安装，无法安装 Subtitle-Translator" >&2
  exit 1
fi

mkdir -p "$(dirname "$REPO_DIR")"

if [[ -d "$REPO_DIR/.git" ]]; then
  :
else
  git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

if [[ $FORCE -eq 1 ]]; then
  uv tool install -e . --force
else
  uv tool install -e .
fi

uv tool update-shell || true
translate --help >/dev/null
echo "installed_repo=$REPO_DIR"
echo "verified=translate --help"
