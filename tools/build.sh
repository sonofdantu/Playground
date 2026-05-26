#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="Release"
CLEAN=0
RUN_TESTS=0
ORT_GENAI=0
ORT_GENAI_ROOT=""
ORT_RUNTIME_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      BUILD_TYPE="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --test)
      RUN_TESTS=1
      shift
      ;;
    --ort-genai)
      ORT_GENAI=1
      shift
      ;;
    --ort-genai-root)
      ORT_GENAI_ROOT="$2"
      shift 2
      ;;
    --ort-runtime-root)
      ORT_RUNTIME_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: tools/build.sh [options]

Options:
  --config <name>            CMake build type, default Release
  --clean                    Remove build directory first
  --test                     Run CTest after building
  --ort-genai                Enable ONNX Runtime GenAI backend
  --ort-genai-root <path>    ORT GenAI SDK root
  --ort-runtime-root <path>  ONNX Runtime SDK root
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
BUILD_DIR="$REPO_ROOT/build"

find_tool() {
  local name="$1"
  local fallback="$REPO_ROOT/.venv/bin/$name"

  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
  elif [[ -x "$fallback" ]]; then
    printf "%s\n" "$fallback"
  else
    return 1
  fi
}

if ! CMAKE_BIN="$(find_tool cmake)"; then
  echo "error: cmake is required" >&2
  echo "Install it system-wide or run ./tools/setup_linux_env.sh to install local build tools under .venv." >&2
  exit 1
fi

if ! CTEST_BIN="$(find_tool ctest)"; then
  CTEST_BIN=""
fi

if NINJA_BIN="$(find_tool ninja)"; then
  HAS_NINJA=1
else
  HAS_NINJA=0
fi

if [[ "$ORT_GENAI" == "1" ]]; then
  ORT_GENAI_ROOT="${ORT_GENAI_ROOT:-$REPO_ROOT/.deps/onnxruntime-genai-0.13.1-linux-x64/onnxruntime-genai-0.13.1-linux-x64}"
  ORT_RUNTIME_ROOT="${ORT_RUNTIME_ROOT:-$REPO_ROOT/.deps/onnxruntime-linux-x64-1.25.1/onnxruntime-linux-x64-1.25.1}"
  ORT_GENAI_ROOT="$(realpath "$ORT_GENAI_ROOT")"
  ORT_RUNTIME_ROOT="$(realpath "$ORT_RUNTIME_ROOT")"
fi

if [[ "$CLEAN" == "1" && -d "$BUILD_DIR" ]]; then
  resolved_build="$(realpath "$BUILD_DIR")"
  resolved_root="$(realpath "$REPO_ROOT")"
  case "$resolved_build" in
    "$resolved_root"/*) rm -rf "$resolved_build" ;;
    *) echo "error: refusing to remove build directory outside repo: $resolved_build" >&2; exit 1 ;;
  esac
fi

configure_args=(
  -S "$REPO_ROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DSCENE_DESC_BUILD_TESTS=ON
)

if [[ "$HAS_NINJA" == "1" ]]; then
  configure_args+=(-G Ninja "-DCMAKE_MAKE_PROGRAM=$NINJA_BIN")
fi

if [[ "$ORT_GENAI" == "1" ]]; then
  if [[ ! -f "$ORT_GENAI_ROOT/include/ort_genai.h" ]]; then
    echo "error: missing ORT GenAI include root: $ORT_GENAI_ROOT/include/ort_genai.h" >&2
    exit 1
  fi
  if ! compgen -G "$ORT_GENAI_ROOT/lib/libonnxruntime-genai.so*" >/dev/null; then
    echo "error: missing ORT GenAI library: $ORT_GENAI_ROOT/lib/libonnxruntime-genai.so*" >&2
    exit 1
  fi
  if [[ -n "$ORT_RUNTIME_ROOT" ]] && ! compgen -G "$ORT_RUNTIME_ROOT/lib/libonnxruntime.so*" >/dev/null; then
    echo "error: missing ONNX Runtime library: $ORT_RUNTIME_ROOT/lib/libonnxruntime.so*" >&2
    exit 1
  fi
  configure_args+=(
    -DSCENE_DESC_ENABLE_ORT_GENAI=ON
    "-DOnnxRuntimeGenAI_ROOT=$ORT_GENAI_ROOT"
    "-DCMAKE_INSTALL_RPATH=\$ORIGIN"
    "-DCMAKE_BUILD_RPATH=\$ORIGIN"
  )
fi

"$CMAKE_BIN" "${configure_args[@]}"
"$CMAKE_BIN" --build "$BUILD_DIR" --config "$BUILD_TYPE"

if [[ "$ORT_GENAI" == "1" ]]; then
  cp -P "$ORT_GENAI_ROOT"/lib/libonnxruntime-genai*.so* "$BUILD_DIR"/
  if [[ -n "$ORT_RUNTIME_ROOT" ]]; then
    cp -P "$ORT_RUNTIME_ROOT"/lib/libonnxruntime.so* "$BUILD_DIR"/
    if compgen -G "$ORT_RUNTIME_ROOT/lib/libonnxruntime_providers_*.so*" >/dev/null; then
      cp -P "$ORT_RUNTIME_ROOT"/lib/libonnxruntime_providers_*.so* "$BUILD_DIR"/
    fi
  fi
  export LD_LIBRARY_PATH="$BUILD_DIR:$ORT_GENAI_ROOT/lib:${ORT_RUNTIME_ROOT:+$ORT_RUNTIME_ROOT/lib:}${LD_LIBRARY_PATH:-}"
fi

if [[ "$RUN_TESTS" == "1" ]]; then
  if [[ -z "$CTEST_BIN" ]]; then
    echo "error: ctest is required to run tests" >&2
    echo "Install it system-wide or run ./tools/setup_linux_env.sh to install local build tools under .venv." >&2
    exit 1
  fi
  "$CTEST_BIN" --test-dir "$BUILD_DIR" --output-on-failure
fi
