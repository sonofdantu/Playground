param(
  [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
  [string]$Config = "Release",

  [switch]$Clean,
  [switch]$Test,
  [switch]$OrtGenAI,
  [string]$OrtGenAIRoot = "",
  [string]$OrtRuntimeRoot = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")
$buildDir = Join-Path $repoRoot "build"
$vsRoot = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
$vcvars = Join-Path $vsRoot "VC\Auxiliary\Build\vcvars64.bat"
$cmake = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninjaDir = Join-Path $vsRoot "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
$runtimePathEntries = @($ninjaDir)

if (-not (Test-Path -LiteralPath $vcvars)) {
  throw "Could not find vcvars64.bat at $vcvars"
}
if (-not (Test-Path -LiteralPath $cmake)) {
  throw "Could not find Visual Studio CMake at $cmake"
}
if (-not (Test-Path -LiteralPath (Join-Path $ninjaDir "ninja.exe"))) {
  throw "Could not find Visual Studio Ninja at $ninjaDir"
}

if ($Clean -and (Test-Path -LiteralPath $buildDir)) {
  $resolvedBuild = (Resolve-Path -LiteralPath $buildDir).Path
  $resolvedRoot = (Resolve-Path -LiteralPath $repoRoot).Path
  if (-not $resolvedBuild.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to remove build directory outside repo: $resolvedBuild"
  }
  Remove-Item -LiteralPath $resolvedBuild -Recurse -Force
}

$configureArgs = @(
  "-S", "`"$repoRoot`"",
  "-B", "`"$buildDir`"",
  "-G", "Ninja",
  "-DSCENE_DESC_BUILD_TESTS=ON"
)

if ($OrtGenAI) {
  $configureArgs += "-DSCENE_DESC_ENABLE_ORT_GENAI=ON"
  if ($OrtGenAIRoot) {
    $configureArgs += "-DOnnxRuntimeGenAI_ROOT=`"$OrtGenAIRoot`""
    $ortGenAILib = Join-Path $OrtGenAIRoot "lib"
    if (Test-Path -LiteralPath $ortGenAILib) {
      $runtimePathEntries += $ortGenAILib
    }
  }
}

$configureCommand = "`"$cmake`" $($configureArgs -join ' ')"
$buildCommand = "`"$cmake`" --build `"$buildDir`" --config $Config"
$pathCommand = "set `"PATH=$($runtimePathEntries -join ';');!PATH!`""

$command = "`"$vcvars`" && $pathCommand && $configureCommand && $buildCommand"

cmd.exe /v:on /s /c $command
if ($LASTEXITCODE -ne 0) {
  exit $LASTEXITCODE
}

if ($OrtGenAI) {
  if (-not $OrtGenAIRoot) {
    throw "-OrtGenAIRoot is required when -OrtGenAI is set"
  }

  $genAiDll = Join-Path $OrtGenAIRoot "lib\onnxruntime-genai.dll"
  if (-not (Test-Path -LiteralPath $genAiDll)) {
    throw "Could not find $genAiDll"
  }
  Copy-Item -LiteralPath $genAiDll -Destination (Join-Path $buildDir "onnxruntime-genai.dll") -Force

  if ($OrtRuntimeRoot) {
    $ortDll = Join-Path $OrtRuntimeRoot "lib\onnxruntime.dll"
    if (-not (Test-Path -LiteralPath $ortDll)) {
      throw "Could not find $ortDll"
    }
    Copy-Item -LiteralPath $ortDll -Destination (Join-Path $buildDir "onnxruntime.dll") -Force
  } else {
    Write-Warning "No -OrtRuntimeRoot provided. Windows may load an incompatible onnxruntime.dll from System32."
  }
}

if ($Test) {
  $testPathEntries = @($buildDir) + $runtimePathEntries
  $testPathCommand = "set `"PATH=$($testPathEntries -join ';');!PATH!`""
  $testCommand = "`"$vcvars`" && $testPathCommand && `"$cmake`" -E chdir `"$buildDir`" ctest --output-on-failure"
  cmd.exe /v:on /s /c $testCommand
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}
