param(
  [string]$OutputDir = "dist\scene-describer",
  [string]$BuildDir = "build",
  [string]$ConfigPath = "configs\qwen3-vl-2b.ini",
  [string]$ModelDir = "",
  [switch]$IncludeModel,
  [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
  param([string]$PathValue, [string]$Label)
  if (-not (Test-Path -LiteralPath $PathValue)) {
    throw "$Label does not exist: $PathValue"
  }
  return (Resolve-Path -LiteralPath $PathValue).Path
}

function Join-RepoPath {
  param([string]$PathValue)
  if ([System.IO.Path]::IsPathRooted($PathValue)) {
    return $PathValue
  }
  return Join-Path $repoRoot $PathValue
}

function Ensure-ChildPath {
  param([string]$Parent, [string]$Child, [string]$Label)
  $parentFull = [System.IO.Path]::GetFullPath($Parent)
  $childFull = [System.IO.Path]::GetFullPath($Child)
  if (-not $childFull.StartsWith($parentFull, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "$Label is outside expected directory: $childFull"
  }
  return $childFull
}

function Copy-IfExists {
  param([string]$Source, [string]$DestinationDir, [System.Collections.ArrayList]$Copied)
  if (Test-Path -LiteralPath $Source) {
    Copy-Item -LiteralPath $Source -Destination $DestinationDir -Force
    [void]$Copied.Add((Split-Path -Leaf $Source))
  }
}

function Set-ConfigModelDir {
  param([string]$Text, [string]$Value)
  $lines = $Text -split "\r?\n"
  $updated = $false
  for ($index = 0; $index -lt $lines.Count; $index++) {
    if ($lines[$index] -match "^\s*model_dir\s*=") {
      $lines[$index] = "model_dir=$Value"
      $updated = $true
    }
  }
  if (-not $updated) {
    $lines += "model_dir=$Value"
  }
  return ($lines -join "`r`n").TrimEnd() + "`r`n"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$resolvedOutput = [System.IO.Path]::GetFullPath((Join-RepoPath $OutputDir))
$resolvedBuild = Resolve-ExistingPath (Join-RepoPath $BuildDir) "Build directory"
$resolvedConfig = Resolve-ExistingPath (Join-RepoPath $ConfigPath) "Config"
$resolvedOutput = Ensure-ChildPath $repoRoot $resolvedOutput "Output directory"

$sceneExe = Join-Path $resolvedBuild "scene_describer.exe"
if (-not (Test-Path -LiteralPath $sceneExe)) {
  throw "Missing scene_describer.exe. Build the project before packaging."
}

if ((Test-Path -LiteralPath $resolvedOutput) -and $Force) {
  $checkedOutput = Ensure-ChildPath $repoRoot (Resolve-Path -LiteralPath $resolvedOutput).Path "Output directory"
  Remove-Item -LiteralPath $checkedOutput -Recurse -Force
}

New-Item -ItemType Directory -Force -Path $resolvedOutput | Out-Null
$binDir = Join-Path $resolvedOutput "bin"
$configDir = Join-Path $resolvedOutput "config"
$docsDir = Join-Path $resolvedOutput "docs"
New-Item -ItemType Directory -Force -Path $binDir, $configDir, $docsDir | Out-Null

$executables = [System.Collections.ArrayList]::new()
Copy-IfExists (Join-Path $resolvedBuild "scene_describer.exe") $binDir $executables
Copy-IfExists (Join-Path $resolvedBuild "scene_analyzer.exe") $binDir $executables
Copy-IfExists (Join-Path $resolvedBuild "scene_describer_benchmark.exe") $binDir $executables
Copy-IfExists (Join-Path $resolvedBuild "scene_model_probe.exe") $binDir $executables

$dlls = [System.Collections.ArrayList]::new()
Copy-IfExists (Join-Path $resolvedBuild "onnxruntime.dll") $binDir $dlls
Copy-IfExists (Join-Path $resolvedBuild "onnxruntime-genai.dll") $binDir $dlls

$runtimeConfig = Join-Path $configDir "runtime.ini"
$configText = Get-Content -LiteralPath $resolvedConfig -Raw
$promptSourceDir = Join-Path $repoRoot "prompts\analyzer"
$promptTargetDir = Join-Path $resolvedOutput "prompts\analyzer"
if (Test-Path -LiteralPath $promptSourceDir) {
  New-Item -ItemType Directory -Force -Path $promptTargetDir | Out-Null
  Get-ChildItem -LiteralPath $promptSourceDir -File | Copy-Item -Destination $promptTargetDir -Force
}
Copy-IfExists (Join-Path $repoRoot "docs\LICENSES.md") $docsDir ([System.Collections.ArrayList]::new()) | Out-Null
Copy-IfExists (Join-Path $repoRoot "docs\MODEL_PROVENANCE.md") $docsDir ([System.Collections.ArrayList]::new()) | Out-Null
Copy-IfExists (Join-Path $repoRoot "docs\BENCHMARKING.md") $docsDir ([System.Collections.ArrayList]::new()) | Out-Null

$resolvedModel = $null
if ($ModelDir) {
  $resolvedModel = Resolve-ExistingPath (Join-RepoPath $ModelDir) "Model directory"
  $provenance = Join-Path $resolvedModel "MODEL_PROVENANCE.json"
  if (Test-Path -LiteralPath $provenance) {
    Copy-Item -LiteralPath $provenance -Destination (Join-Path $docsDir "MODEL_PROVENANCE.local.json") -Force
  }

  if ($IncludeModel) {
    $modelTarget = Join-Path $resolvedOutput "model"
    New-Item -ItemType Directory -Force -Path $modelTarget | Out-Null
    Get-ChildItem -LiteralPath $resolvedModel -Force | Copy-Item -Destination $modelTarget -Recurse -Force
    $configText = Set-ConfigModelDir $configText "model"
  } else {
    $configText = Set-ConfigModelDir $configText $resolvedModel
  }
}

$configText | Set-Content -LiteralPath $runtimeConfig -Encoding ASCII

$runScript = @'
param(
  [string]$Image,
  [int]$MaxNewTokens = 48
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$imagePath = if ([System.IO.Path]::IsPathRooted($Image)) {
  $Image
} else {
  [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Image))
}
$exe = Join-Path $root "bin\scene_describer.exe"
$config = Join-Path $root "config\runtime.ini"
Push-Location $root
try {
  & $exe --config $config --image $imagePath --max-new-tokens $MaxNewTokens --json
} finally {
  Pop-Location
}
'@
$runScript | Set-Content -LiteralPath (Join-Path $resolvedOutput "run.ps1") -Encoding UTF8

$runAnalyzerScript = @'
param(
  [string]$Image,
  [string]$RequestId = "package-smoke",
  [int64]$TimestampMs = 0,
  [int]$MaxNewTokens = 64
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$imagePath = if ([System.IO.Path]::IsPathRooted($Image)) {
  $Image
} else {
  [System.IO.Path]::GetFullPath((Join-Path (Get-Location) $Image))
}
$exe = Join-Path $root "bin\scene_analyzer.exe"
$config = Join-Path $root "config\runtime.ini"
$templates = Join-Path $root "prompts\analyzer"
Push-Location $root
try {
  & $exe --config $config --templates $templates --image $imagePath --timestamp-ms $TimestampMs --request-id $RequestId --max-new-tokens $MaxNewTokens --json
} finally {
  Pop-Location
}
'@
$runAnalyzerScript | Set-Content -LiteralPath (Join-Path $resolvedOutput "run_analyzer.ps1") -Encoding UTF8

$manifest = [ordered]@{
  created_utc = (Get-Date).ToUniversalTime().ToString("o")
  build_dir = $resolvedBuild
  config = "config/runtime.ini"
  model_dir = $resolvedModel
  model_included = [bool]$IncludeModel
  executables = @($executables)
  dlls = @($dlls)
  notes = @(
    "Model files are copied only when -IncludeModel is set.",
    "For production, require MODEL_PROVENANCE.json classification production_candidate or production before shipping."
  )
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resolvedOutput "package_manifest.json") -Encoding UTF8

Write-Host "packaged runtime: $resolvedOutput"
Write-Host "executables: $($executables -join ', ')"
Write-Host "dlls: $($dlls -join ', ')"
if ($resolvedModel -and -not $IncludeModel) {
  Write-Host "model referenced but not copied: $resolvedModel"
}
