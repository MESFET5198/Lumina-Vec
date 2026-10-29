#!/usr/bin/env bash
# Real-ESRGAN anime video model (BSD-3-Clause, xinntao/Real-ESRGAN)
set -e
mkdir -p "$(dirname "$0")/../weights"
curl -L -o "$(dirname "$0")/../weights/realesr-animevideov3.pth" \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth
