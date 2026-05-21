param(
  [string]$Python = "python",
  [string]$VenvPath = ".venv",
  [string]$OrtGenAIVersion = "0.13.1",
  [switch]$SkipPackageInstall,
  [switch]$SkipSourceFetch
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$venvRoot = Join-Path $repoRoot $VenvPath
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$requirements = Join-Path $repoRoot "tools\export-requirements.txt"
$sourceRoot = Join-Path $repoRoot ".deps\onnxruntime-genai-src-v$OrtGenAIVersion"
$builder = Join-Path $sourceRoot "src\python\py\models\builder.py"

if (-not (Test-Path -LiteralPath $venvPython)) {
  Write-Host "Creating Python virtual environment at $venvRoot"
  & $Python -m venv $venvRoot
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if (-not $SkipPackageInstall) {
  & $venvPython -m pip install -r $requirements
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

if (-not $SkipSourceFetch -and -not (Test-Path -LiteralPath $builder)) {
  New-Item -ItemType Directory -Force -Path (Join-Path $repoRoot ".deps") | Out-Null
  git clone --depth 1 --branch "v$OrtGenAIVersion" https://github.com/microsoft/onnxruntime-genai.git $sourceRoot
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

if (-not (Test-Path -LiteralPath $builder)) {
  throw "Could not find ORT GenAI model builder at $builder"
}

Write-Host ""
Write-Host "Python: $venvPython"
Write-Host "Builder: $builder"
Write-Host ""
Write-Host "Next command:"
Write-Host ".\tools\export_model.ps1 -ModelName Qwen/Qwen3.5-2B -OutputDir models\qwen3.5-2b -Precision int4 -ExecutionProvider cpu"
