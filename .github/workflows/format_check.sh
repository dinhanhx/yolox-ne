#!/bin/bash -e

set -e

ruff check yolox exps tools tests
echo "All ruff checks passed!"
