#!/usr/bin/env python3
"""Fast repository sanity checks that do not require a C++ toolchain."""

from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = [
    "CMakeLists.txt",
    "include/scene_describer/analyzer.hpp",
    "include/scene_describer/scene_describer.hpp",
    "src/analyzer/analyzer.cpp",
    "src/analyzer_cli/main.cpp",
    "src/benchmarks/benchmark_main.cpp",
    "src/cli/main.cpp",
    "src/backends/mock_backend.cpp",
    "src/backends/ort_genai_backend.cpp",
    "src/tools/model_probe.cpp",
    "configs/mock.local.ini",
    "configs/qwen3.5-2b-onnxopt-cuda.ini",
    "configs/qwen3.5-2b-onnxopt.ini",
    "docs/ARCHITECTURE.md",
    "docs/ANALYZER.md",
    "docs/BENCHMARKING.md",
    "docs/CUDA.md",
    "docs/QUICKSTART.md",
    "docs/EDGE_ROADMAP.md",
    "docs/LINUX_REPRODUCTION.md",
    "docs/MODEL_EXPORT.md",
    "docs/MODEL_READINESS.md",
    "docs/PACKAGING.md",
    "docs/MODEL_PROVENANCE.md",
    "docs/RUNTIME_ASSETS.md",
    "docs/PROGRESS.md",
    "PROJECT_STATE.md",
    "tools/benchmark_cli.py",
    "tools/benchmark_cuda_frame_batch.sh",
    "tools/build.sh",
    "tools/cuda_library_path.sh",
    "tools/debug_cuda_ort_genai.sh",
    "tools/export_model.ps1",
    "tools/fetch_hf_oga_package.py",
    "tools/fetch_runtime_deps.sh",
    "tools/fetch_runtime_deps.ps1",
    "tools/make_test_image.py",
    "tools/make_detail_crops.py",
    "tools/make_test_frames.py",
    "tools/model_provenance.py",
    "tools/model_readiness.py",
    "tools/package_runtime.ps1",
    "tools/prepare_qwen35_onnxopt_genai.py",
    "tools/setup_export_env.sh",
    "tools/setup_export_env.ps1",
    "tools/setup_linux_cuda_env.sh",
    "tools/setup_linux_env.sh",
    "tools/smoke_ort_genai.sh",
    "tools/smoke_cuda_frame_batch.sh",
    "tools/smoke_ort_genai.ps1",
    "tools/validate_model_package.py",
    "prompts/analyzer/base_guardrails.txt",
    "prompts/analyzer/task_rules.txt",
    "prompts/analyzer/local_batch.txt",
    "prompts/analyzer/history_context.txt",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not Path(path).exists()]
    if missing:
        raise SystemExit("missing required files:\n" + "\n".join(missing))

    for config in Path("configs").glob("*.ini"):
        for line_number, line in enumerate(config.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                raise SystemExit(f"{config}:{line_number}: expected key=value")

    print("project sanity checks passed")


if __name__ == "__main__":
    main()
