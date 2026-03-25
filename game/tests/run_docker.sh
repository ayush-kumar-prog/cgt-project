#!/bin/bash
# Run leader tests against real engine via Docker
# Usage: ./run_docker.sh [path_to_leaders.py]
set -e
cd "$(dirname "$0")/../.."
LEADERS="${1:-game/src/leaders.py}"
PROJ="game/extracted/COMP34612_Student/project_files"

docker run --rm --platform linux/amd64 \
  -v "$(pwd)/$PROJ:/app" \
  -v "$(pwd)/$LEADERS:/app/leaders_code.py" \
  -v "$(pwd)/game/extracted/COMP34612_Student/test_harness.py:/app/test_harness.py" \
  -w /app python:3.12-slim bash -c \
  "pip install openpyxl xlsxwriter pandas numpy -q 2>/dev/null && python3 test_harness.py"
