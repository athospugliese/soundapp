#!/usr/bin/env bash
# Baixa os modelos pré-empacotados do GitHub Release e extrai em models/.
# Uso: bash download_models.sh
set -euo pipefail

REPO="athospugliese/soundapp"
TAG="models-v1"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/models"
mkdir -p "$DIR"

# Cada entrada: <arquivo.tar>  <pasta-resultante-em-models/>
ASSETS=(
  "faster-whisper-small.tar   faster-whisper-small"
  "faster-whisper-medium.tar  faster-whisper-medium"
  "nemo.tar                   nemo"
)

for entry in "${ASSETS[@]}"; do
  tar_name=$(echo "$entry" | awk '{print $1}')
  out_dir=$(echo "$entry" | awk '{print $2}')
  if [ -d "$DIR/$out_dir" ]; then
    echo "✓ $out_dir já existe — pulando"
    continue
  fi
  echo "⬇️  baixando $tar_name ..."
  curl -L --fail --progress-bar "$BASE/$tar_name" -o "/tmp/$tar_name"
  echo "📦 extraindo em models/$out_dir ..."
  tar -xf "/tmp/$tar_name" -C "$DIR"
  rm -f "/tmp/$tar_name"
done

echo "✅ Modelos prontos em $DIR"
