param(
  [string]$ModelName = "Qwen/Qwen3.5-2B",
  [string]$OutputDir = "models\qwen3.5-2b",

  [ValidateSet("int4", "bf16", "fp16", "fp32")]
  [string]$Precision = "int4",

  [ValidateSet("cpu", "cuda", "dml", "webgpu", "NvTensorRtRtx")]
  [string]$ExecutionProvider = "cpu",

  [string]$CacheDir = ".cache\huggingface",
  [string]$VenvPath = ".venv",
  [string]$OrtGenAISource = ".deps\onnxruntime-genai-src-v0.13.1",
  [string[]]$ExtraOptions = @("hf_remote=true", "hf_token=false"),
  [switch]$ConfigOnly,
  [switch]$AllowTextOnly
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$venvPython = Join-Path $repoRoot "$VenvPath\Scripts\python.exe"
$sourceRoot = Join-Path $repoRoot $OrtGenAISource
$builder = Join-Path $sourceRoot "src\python\py\models\builder.py"
$builderDir = Split-Path -Parent $builder
$resolvedOutput = Join-Path $repoRoot $OutputDir
$resolvedCache = Join-Path $repoRoot $CacheDir

if (-not (Test-Path -LiteralPath $venvPython)) {
  throw "Python environment not found at $venvPython. Run tools\setup_export_env.ps1 first."
}
if (-not (Test-Path -LiteralPath $builder)) {
  throw "ORT GenAI model builder not found at $builder. Run tools\setup_export_env.ps1 first."
}
if (Test-Path -LiteralPath $resolvedOutput) {
  Write-Warning "Output directory already exists: $resolvedOutput"
}

New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
New-Item -ItemType Directory -Force -Path $resolvedCache | Out-Null

if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
  $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
}

$builderArgs = @(
  $builder,
  "-m", $ModelName,
  "-o", $resolvedOutput,
  "-p", $Precision,
  "-e", $ExecutionProvider,
  "-c", $resolvedCache
)

$options = @($ExtraOptions)
if ($ConfigOnly) {
  $options += "config_only=true"
}
if ($options.Count -gt 0) {
  $builderArgs += "--extra_options"
  $builderArgs += $options
}

Write-Host "Running ORT GenAI model builder from $builder"
Write-Host "Model: $ModelName"
Write-Host "Output: $resolvedOutput"
Write-Host "Precision: $Precision"
Write-Host "Execution provider: $ExecutionProvider"

Push-Location $builderDir
try {
  & $venvPython @builderArgs
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
} finally {
  Pop-Location
}

Write-Host ""
Write-Host "Validating generated package"
$validationArgs = @((Join-Path $repoRoot "tools\validate_model_package.py"), $resolvedOutput)
if ($ConfigOnly) {
  $validationArgs += "--allow-config-only"
  $validationArgs += "--allow-decoder-only"
}
if ($AllowTextOnly) {
  $validationArgs += "--allow-decoder-only"
}
if (-not $ConfigOnly -and -not $AllowTextOnly) {
  $validationArgs += "--require-multimodal"
}
& $venvPython @validationArgs
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}
