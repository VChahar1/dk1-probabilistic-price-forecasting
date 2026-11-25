#!/usr/bin/env bash
set -euo pipefail
: "${ENTSOE_API_KEY:?Export your own ENTSOE_API_KEY before downloading}"
.venv/bin/python -m src.data --zone DK_1 --start 2019-01-01 --end 2026-09-01
.venv/bin/python -m src.features --zone DK_1
.venv/bin/python -m src.weather --start 2024-03-14 --end 2026-08-30
.venv/bin/python -m src.weather --build
