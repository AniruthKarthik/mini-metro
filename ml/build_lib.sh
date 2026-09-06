#!/bin/bash
set -e

echo "Building Mini Metro shared library..."
cd ../simulator
go build -buildmode=c-shared -o ../ml/libminimetro.so ./c_api
echo "Done. Created ../ml/libminimetro.so"
