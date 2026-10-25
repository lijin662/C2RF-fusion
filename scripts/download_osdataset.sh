#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p downloads
if [[ -s downloads/OSdataset.zip ]]; then
  "$PYTHON" -c 'from zipfile import ZipFile; z=ZipFile("downloads/OSdataset.zip"); assert z.testzip() is None; print("Existing ZIP: CRC PASS")'
  exit 0
fi
"$PYTHON" -m gdown 'https://drive.google.com/uc?id=10gySGfj0-pH1N2TeZWYU1H38OskzKC27' \
  --output downloads/OSdataset.zip.part --continue
"$PYTHON" -c 'from zipfile import ZipFile; z=ZipFile("downloads/OSdataset.zip.part"); assert z.testzip() is None; print("CRC PASS")'
mv downloads/OSdataset.zip.part downloads/OSdataset.zip
