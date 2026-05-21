# Runtime Packaging

`tools\package_runtime.ps1` creates a local deployment folder for target-device smoke tests. It copies:

- `scene_describer.exe`
- `scene_analyzer.exe`
- `scene_describer_benchmark.exe` when available
- `scene_model_probe.exe` when available
- app-local `onnxruntime.dll` and `onnxruntime-genai.dll` when available
- a selected runtime config as `config\runtime.ini`
- analyzer prompt templates under `prompts\analyzer`
- provenance and benchmark docs
- model provenance JSON when `-ModelDir` is supplied

Model binaries are not copied by default because they are large and need explicit license/provenance handling. When `-ModelDir` is supplied without `-IncludeModel`, the packaged config points at the resolved local model directory. Use `-IncludeModel` only for an internal device bundle after the package classification is acceptable for that use; in that mode the packaged config points at `model` inside the bundle.

## Command

```powershell
.\tools\package_runtime.ps1 `
  -OutputDir dist\scene-describer `
  -BuildDir build `
  -ConfigPath configs\qwen3.5-2b-onnxopt.ini `
  -ModelDir models\qwen3.5-2b-onnxopt-q4f16 `
  -Force
```

The generated package includes `run.ps1`:

```powershell
.\dist\scene-describer\run.ps1 -Image tmp\smoke.png -MaxNewTokens 48
```

It also includes `run_analyzer.ps1`:

```powershell
.\dist\scene-describer\run_analyzer.ps1 -Image tmp\smoke.png -RequestId package-smoke -TimestampMs 1000
```

For production packaging, require:

- `MODEL_PROVENANCE.json` with `classification` set to `production_candidate` or `production`.
- Explicit commercial license approval for the model source and any prebuilt ONNX package source.
- Target execution-provider DLLs copied from the exact runtime build used in validation.
