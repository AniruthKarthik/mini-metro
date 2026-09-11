#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Building Mini Metro shared library..."
cd "$SCRIPT_DIR/../simulator"
go build -buildmode=c-shared -o "$SCRIPT_DIR/libminimetro.so" ./c_api
echo "Done. Created $SCRIPT_DIR/libminimetro.so"
