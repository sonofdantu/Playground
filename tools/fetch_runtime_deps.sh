#!/usr/bin/env bash
set -euo pipefail

ORT_GENAI_VERSION="${ORT_GENAI_VERSION:-0.13.1}"
ORT_RUNTIME_VERSION="${ORT_RUNTIME_VERSION:-1.25.1}"
ORT_GENAI_FLAVOR="${ORT_GENAI_FLAVOR:-linux-x64}"
ORT_RUNTIME_FLAVOR="${ORT_RUNTIME_FLAVOR:-linux-x64}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      ORT_GENAI_FLAVOR="linux-x64"
      ORT_RUNTIME_FLAVOR="linux-x64"
      shift
      ;;
    --cuda)
      ORT_GENAI_FLAVOR="linux-x64-cuda"
      ORT_RUNTIME_FLAVOR="linux-x64-gpu"
      shift
      ;;
    --cuda13)
      ORT_GENAI_FLAVOR="linux-x64-cuda"
      ORT_RUNTIME_FLAVOR="linux-x64-gpu_cuda13"
      shift
      ;;
    --ort-genai-version)
      ORT_GENAI_VERSION="$2"
      shift 2
      ;;
    --ort-runtime-version)
      ORT_RUNTIME_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/fetch_runtime_deps.sh [options]

Options:
  --cpu                       Fetch Linux CPU packages, default
  --cuda                      Fetch Linux CUDA packages using the CUDA 12 ORT GPU runtime
  --cuda13                    Fetch Linux CUDA packages using the CUDA 13 ORT GPU runtime
  --ort-genai-version <ver>   ORT GenAI version, default 0.13.1
  --ort-runtime-version <ver> ONNX Runtime version, default 1.25.1
EOF
      exit 0
      ;;
    *)
      echo "error: unknown option $1" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DEPS_DIR="$REPO_ROOT/.deps"

mkdir -p "$DEPS_DIR"

download_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ -f "$destination" ]]; then
    echo "Using existing $destination"
    return
  fi

  echo "Downloading $url"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$destination"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$destination" "$url"
  else
    echo "error: curl or wget is required" >&2
    exit 1
  fi
}

extract_if_missing() {
  local archive="$1"
  local destination="$2"
  if [[ -d "$destination" ]]; then
    echo "Using existing $destination"
    return
  fi

  echo "Extracting $archive"
  mkdir -p "$destination"
  tar -xzf "$archive" -C "$destination"
}

resolve_root() {
  local extract_dir="$1"
  local package_name="$2"
  local marker="$3"

  if [[ -f "$extract_dir/$package_name/$marker" ]]; then
    printf "%s\n" "$extract_dir/$package_name"
  elif [[ -f "$extract_dir/$marker" ]]; then
    printf "%s\n" "$extract_dir"
  else
    local found
    found="$(find "$extract_dir" -mindepth 1 -maxdepth 2 -type f -path "*/$marker" -print -quit)"
    if [[ -n "$found" ]]; then
      printf "%s\n" "${found%/$marker}"
    else
      echo "error: could not find expected marker $marker under $extract_dir" >&2
      exit 1
    fi
  fi
}

ORT_GENAI_NAME="onnxruntime-genai-$ORT_GENAI_VERSION-$ORT_GENAI_FLAVOR"
ORT_GENAI_ARCHIVE="$DEPS_DIR/$ORT_GENAI_NAME.tar.gz"
ORT_GENAI_EXTRACT="$DEPS_DIR/$ORT_GENAI_NAME"
ORT_GENAI_URL="https://github.com/microsoft/onnxruntime-genai/releases/download/v$ORT_GENAI_VERSION/$ORT_GENAI_NAME.tar.gz"

ORT_RUNTIME_NAME="onnxruntime-$ORT_RUNTIME_FLAVOR-$ORT_RUNTIME_VERSION"
ORT_RUNTIME_ARCHIVE="$DEPS_DIR/$ORT_RUNTIME_NAME.tgz"
ORT_RUNTIME_EXTRACT="$DEPS_DIR/$ORT_RUNTIME_NAME"
ORT_RUNTIME_URL="https://github.com/microsoft/onnxruntime/releases/download/v$ORT_RUNTIME_VERSION/$ORT_RUNTIME_NAME.tgz"

download_if_missing "$ORT_GENAI_URL" "$ORT_GENAI_ARCHIVE"
extract_if_missing "$ORT_GENAI_ARCHIVE" "$ORT_GENAI_EXTRACT"

download_if_missing "$ORT_RUNTIME_URL" "$ORT_RUNTIME_ARCHIVE"
extract_if_missing "$ORT_RUNTIME_ARCHIVE" "$ORT_RUNTIME_EXTRACT"

ORT_GENAI_ROOT="$(resolve_root "$ORT_GENAI_EXTRACT" "$ORT_GENAI_NAME" "include/ort_genai.h")"
ORT_RUNTIME_ROOT="$(resolve_root "$ORT_RUNTIME_EXTRACT" "$ORT_RUNTIME_NAME" "lib/libonnxruntime.so")"

echo
echo "ORT GenAI root: $ORT_GENAI_ROOT"
echo "ORT Runtime root: $ORT_RUNTIME_ROOT"
echo
if [[ "$ORT_GENAI_FLAVOR" == *cuda* || "$ORT_RUNTIME_FLAVOR" == *gpu* ]]; then
  echo "CUDA user-space libraries:"
  echo "./tools/setup_linux_cuda_env.sh"
  echo
fi
echo "Build command:"
echo "./tools/build.sh --test --ort-genai \\"
echo "  --ort-genai-root \"$ORT_GENAI_ROOT\" \\"
echo "  --ort-runtime-root \"$ORT_RUNTIME_ROOT\""
