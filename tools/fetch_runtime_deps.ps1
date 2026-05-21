param(
  [string]$OrtGenAIVersion = "0.13.1",
  [string]$OrtRuntimeVersion = "1.25.1",

  [ValidateSet("win-x64", "win-x64-dml", "win-x64-cuda", "win-x64-cuda-winml", "win-arm64")]
  [string]$OrtGenAIFlavor = "win-x64",

  [ValidateSet("win-x64")]
  [string]$OrtRuntimeFlavor = "win-x64"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$depsDir = Join-Path $repoRoot ".deps"
New-Item -ItemType Directory -Force -Path $depsDir | Out-Null

function Download-IfMissing {
  param(
    [Parameter(Mandatory = $true)][string]$Url,
    [Parameter(Mandatory = $true)][string]$Destination
  )

  if (Test-Path -LiteralPath $Destination) {
    Write-Host "Using existing $Destination"
    return
  }

  Write-Host "Downloading $Url"
  Invoke-WebRequest -Uri $Url -OutFile $Destination
}

function Expand-IfMissing {
  param(
    [Parameter(Mandatory = $true)][string]$Archive,
    [Parameter(Mandatory = $true)][string]$Destination
  )

  if (Test-Path -LiteralPath $Destination) {
    Write-Host "Using existing $Destination"
    return
  }

  Write-Host "Expanding $Archive"
  Expand-Archive -LiteralPath $Archive -DestinationPath $Destination
}

$ortGenAIName = "onnxruntime-genai-$OrtGenAIVersion-$OrtGenAIFlavor"
$ortGenAIArchive = Join-Path $depsDir "$ortGenAIName.zip"
$ortGenAIExtract = Join-Path $depsDir $ortGenAIName
$ortGenAIUrl = "https://github.com/microsoft/onnxruntime-genai/releases/download/v$OrtGenAIVersion/$ortGenAIName.zip"

$ortRuntimeName = "onnxruntime-$OrtRuntimeFlavor-$OrtRuntimeVersion"
$ortRuntimeArchive = Join-Path $depsDir "$ortRuntimeName.zip"
$ortRuntimeExtract = Join-Path $depsDir $ortRuntimeName
$ortRuntimeUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtRuntimeVersion/$ortRuntimeName.zip"

Download-IfMissing -Url $ortGenAIUrl -Destination $ortGenAIArchive
Expand-IfMissing -Archive $ortGenAIArchive -Destination $ortGenAIExtract

Download-IfMissing -Url $ortRuntimeUrl -Destination $ortRuntimeArchive
Expand-IfMissing -Archive $ortRuntimeArchive -Destination $ortRuntimeExtract

$ortGenAIRoot = Join-Path $ortGenAIExtract $ortGenAIName
$ortRuntimeRoot = Join-Path $ortRuntimeExtract $ortRuntimeName

if (-not (Test-Path -LiteralPath (Join-Path $ortGenAIRoot "include\ort_genai.h"))) {
  throw "ORT GenAI package did not expand to the expected root: $ortGenAIRoot"
}
if (-not (Test-Path -LiteralPath (Join-Path $ortRuntimeRoot "lib\onnxruntime.dll"))) {
  throw "ONNX Runtime package did not expand to the expected root: $ortRuntimeRoot"
}

Write-Host ""
Write-Host "ORT GenAI root: $ortGenAIRoot"
Write-Host "ORT Runtime root: $ortRuntimeRoot"
Write-Host ""
Write-Host "Build command:"
Write-Host ".\tools\build.ps1 -Clean -Test -OrtGenAI -OrtGenAIRoot `"$ortGenAIRoot`" -OrtRuntimeRoot `"$ortRuntimeRoot`""
