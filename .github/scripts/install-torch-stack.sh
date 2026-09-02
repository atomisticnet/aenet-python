#!/usr/bin/env bash

set -euo pipefail

: "${ENV_NAME:?ENV_NAME must be set}"
: "${TORCH_VERSION:?TORCH_VERSION must be set}"
: "${PYG_WHEEL_URL:?PYG_WHEEL_URL must be set}"

run_in_env=(micromamba run -n "${ENV_NAME}")

"${run_in_env[@]}" python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  "torch==${TORCH_VERSION}"

if ! "${run_in_env[@]}" python -m pip install \
  --only-binary=:all: \
  torch-scatter torch-cluster \
  --find-links "${PYG_WHEEL_URL}"
then
  echo "Prebuilt PyG wheels unavailable; building CPU extensions from source."
  "${run_in_env[@]}" python -m pip install --upgrade setuptools wheel
  "${run_in_env[@]}" env MAX_JOBS=2 python -m pip install \
    --no-build-isolation torch-scatter torch-cluster
fi
