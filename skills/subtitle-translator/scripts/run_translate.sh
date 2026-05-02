#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISCOVER_SCRIPT="$SCRIPT_DIR/discover_subtitle_translator.sh"

if [[ "${1:-}" == "--" ]]; then
  shift
fi

set +e
DISCOVERY_OUTPUT="$(bash "$DISCOVER_SCRIPT" 2>/dev/null)"
STATUS=$?
set -e

if [[ $STATUS -eq 0 ]]; then
  MODE="$(printf '%s\n' "$DISCOVERY_OUTPUT" | awk -F= '/^mode=/{print $2}')"
  REPO="$(printf '%s\n' "$DISCOVERY_OUTPUT" | awk -F= '/^repo=/{print $2}')"

  if [[ "$MODE" == "cli" ]]; then
    exec translate "$@"
  fi

  if [[ "$MODE" == "repo" && -n "$REPO" ]]; then
    cd "$REPO"
    exec uv run python -m subtitle_translator.cli "$@"
  fi
fi

echo "ERROR: Subtitle-Translator 未安装，先运行 scripts/install_subtitle_translator.sh" >&2
exit 1
