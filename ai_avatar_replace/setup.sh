#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

echo ""
echo "Ready. Examples:"
echo "  python main.py my_video.mp4 new_background.jpg --avatar-prompt \"a friendly synthetic host\" -o output/result.mp4"
echo "  python main.py my_video.mp4 new_background.jpg --avatar-image my_character.png -o output/result.mp4"
echo "  python app.py   # web UI"
