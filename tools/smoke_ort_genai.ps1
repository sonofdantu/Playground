param(
  [string]$ConfigPath = "configs\qwen3-vl-2b.ini",
  [string]$ModelDir = "models\qwen3-vl-2b-instruct",
  [string]$ImagePath = "tmp\smoke.png",
  [string]$OrtGenAIRoot = ".deps\onnxruntime-genai-0.13.1-win-x64\onnxruntime-genai-0.13.1-win-x64",
  [string]$OrtRuntimeRoot = ".deps\onnxruntime-win-x64-1.25.1\onnxruntime-win-x64-1.25.1",
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$exe = Join-Path $repoRoot "build\scene_describer.exe"
$resolvedModelDir = Join-Path $repoRoot $ModelDir
$resolvedImagePath = Join-Path $repoRoot $ImagePath

if (-not (Test-Path -LiteralPath $python)) {
  throw "Missing $python. Run tools\setup_export_env.ps1 first."
}
if (-not (Test-Path -LiteralPath $resolvedModelDir)) {
  throw "Missing model package $resolvedModelDir. See docs\MODEL_EXPORT.md smoke-test package instructions."
}

& $python (Join-Path $repoRoot "tools\validate_model_package.py") $resolvedModelDir --require-multimodal --require-provenance
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if (-not (Test-Path -LiteralPath $resolvedImagePath)) {
  & $python (Join-Path $repoRoot "tools\make_test_image.py")
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

if (-not $SkipBuild) {
  & (Join-Path $repoRoot "tools\build.ps1") -Test -OrtGenAI -OrtGenAIRoot $OrtGenAIRoot -OrtRuntimeRoot $OrtRuntimeRoot
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

if (-not (Test-Path -LiteralPath $exe)) {
  throw "Missing $exe"
}

$output = & $exe --config $ConfigPath --model-dir $ModelDir --image $ImagePath --max-new-tokens 48 --json
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

$json = $output | Out-String
$result = $json | ConvertFrom-Json
if (-not $result.text -or $result.text.Trim().Length -eq 0) {
  throw "Smoke run returned empty text: $json"
}
if ($result.text -match "^\s*user\s") {
  throw "Smoke run appears to include the prompt prefix: $($result.text)"
}

Write-Host "ORT GenAI smoke passed:"
Write-Host $result.text
