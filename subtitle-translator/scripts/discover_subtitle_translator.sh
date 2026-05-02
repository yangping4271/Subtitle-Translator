#!/usr/bin/env bash
set -euo pipefail

has_repo_markers() {
  local dir="$1"
  [[ -f "$dir/pyproject.toml" && -f "$dir/src/subtitle_translator/cli.py" ]]
}

if command -v translate >/dev/null 2>&1; then
  echo "mode=cli"
  echo "command=translate"
  exit 0
fi

candidates=(
  "$PWD"
  "$HOME/tools/Subtitle-Translator"
  "$HOME/code/Subtitle-Translator"
  "$HOME/src/Subtitle-Translator"
)

for dir in "${candidates[@]}"; do
  if has_repo_markers "$dir"; then
    echo "mode=repo"
    echo "repo=$dir"
    echo "command=uv run python -m subtitle_translator.cli"
    exit 0
  fi
done

echo "mode=missing"
echo "repo=$HOME/tools/Subtitle-Translator"
exit 1
