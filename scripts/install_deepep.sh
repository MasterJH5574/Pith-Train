#!/usr/bin/env bash
# Idempotent post-`uv sync` setup for DeepEP v2.
#
# Handles: NCCL upgrade past torch's pin, ninja, libnccl SONAME symlink, and
# the editable install of DeepEP at the pinned commit. Multi-arch build for
# Hopper+Blackwell unless TORCH_CUDA_ARCH_LIST is overridden.
#
# Override DEEPEP_DIR to relocate the DeepEP clone (default: $HOME/Workspace/DeepEP).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
PY="${VENV}/bin/python"
DEEPEP_COMMIT="b306af0"
DEEPEP_DIR="${DEEPEP_DIR:-${HOME}/Workspace/DeepEP}"

if ! command -v uv >/dev/null 2>&1; then
    echo "error: uv not found on PATH" >&2
    exit 1
fi
if [[ ! -x "${PY}" ]]; then
    echo "error: ${PY} not found. Run \`uv sync\` first." >&2
    exit 1
fi

echo "[install_deepep] Bumping nvidia-nccl-cu13 past torch's pin (>=2.30.4)..."
uv pip install --no-deps "nvidia-nccl-cu13>=2.30.4"

echo "[install_deepep] Ensuring ninja is installed..."
uv pip install --no-deps ninja

echo "[install_deepep] Locating nvidia-nccl-cu13 lib dir..."
NCCL_LIBDIR="$(${PY} - <<'EOF'
import os
from importlib.metadata import distributions
for dist in distributions():
    name = (dist.metadata.get("Name") or "").lower()
    if "nvidia-nccl" in name or "nvidia_nccl" in name:
        site = str(dist._path.parent)
        path = os.path.join(site, "nvidia", "nccl", "lib")
        assert os.path.isdir(path), f"{path} not a dir"
        print(path)
        break
else:
    raise SystemExit("nvidia-nccl-cu* not installed")
EOF
)"
echo "[install_deepep] NCCL lib dir: ${NCCL_LIBDIR}"

if [[ -e "${NCCL_LIBDIR}/libnccl.so" ]]; then
    echo "[install_deepep] libnccl.so symlink already present."
else
    echo "[install_deepep] Creating libnccl.so -> libnccl.so.2 symlink (DeepEP setup.py bug workaround)."
    ln -s libnccl.so.2 "${NCCL_LIBDIR}/libnccl.so"
fi

echo "[install_deepep] Detected GPU arch (informational):"
"${PY}" - <<'EOF' || true
import torch
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    print(f"  GPU 0: SM{major}{minor} ({major}.{minor})")
else:
    print("  no GPU visible (build will still produce a multi-arch binary)")
EOF
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0 10.0}"
echo "[install_deepep] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

echo "[install_deepep] Cloning/updating DeepEP at ${DEEPEP_DIR}..."
if [[ ! -d "${DEEPEP_DIR}/.git" ]]; then
    git clone https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_DIR}"
fi
(
    cd "${DEEPEP_DIR}"
    git fetch origin
    git checkout --detach "${DEEPEP_COMMIT}"
)

echo "[install_deepep] Editable install of DeepEP into the venv..."
EP_SUPPRESS_NCCL_CHECK=1 \
LIBRARY_PATH="${NCCL_LIBDIR}:${LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    uv pip install --no-deps --no-build-isolation -e "${DEEPEP_DIR}"

echo "[install_deepep] Verifying import..."
EP_SUPPRESS_NCCL_CHECK=1 "${PY}" -c "import deep_ep; print('deep_ep ok:', deep_ep.__version__)"
echo "[install_deepep] Done."
