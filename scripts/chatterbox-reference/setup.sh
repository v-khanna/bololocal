#!/usr/bin/env bash
# scripts/chatterbox-reference/setup.sh
# Creates a Python venv with mlx-audio installed for Chatterbox reference validation.
# Not bundled with Bolo — pure developer tool.
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install mlx-audio huggingface_hub safetensors numpy

echo ""
echo "Done. Activate with:"
echo "  source scripts/chatterbox-reference/venv/bin/activate"
