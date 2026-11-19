#!/bin/sh
# Installe Stockfish 18 (binaire officiel Linux AVX2) dans bin/stockfish
set -e
cd "$(dirname "$0")/.."
mkdir -p bin /tmp/sf18
curl -sSL -o /tmp/sf18/sf.tar https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-avx2.tar
tar xf /tmp/sf18/sf.tar -C /tmp/sf18
cp "$(find /tmp/sf18 -type f -name 'stockfish-ubuntu*' | head -1)" bin/stockfish
chmod +x bin/stockfish
