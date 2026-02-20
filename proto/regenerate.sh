#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I proto --python_out=proto --grpc_python_out=proto proto/*.proto
echo "Proto files regenerated in proto/"
