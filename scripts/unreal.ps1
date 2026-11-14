param(
    [ValidateSet('Probe', 'FleshProbe', 'PrepareBones', 'Capture', 'Open')]
    [string]$Action = 'Probe',
    [string]$EngineRoot = 'C:\Program Files\Epic Games\UE_5.8',
    [switch]$EnableRendering
)

$ErrorActionPreference = 'Stop'
if ($Action -in @('Open', 'Capture') -and -not $EnableRendering) {
    throw 'Rendering is opt-in after a reported system hang. Add -EnableRendering for a deliberate graphics run.'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$integrationRoot = Join-Path $projectRoot 'integrations\unreal'
$projectFile = Join-Path $integrationRoot 'MyoD1\MyoD1.uproject'
$version = Get-Content -Raw (Join-Path $EngineRoot 'Engine\Build\Build.version') | ConvertFrom-Json
$expected = Get-Content -Raw (Join-Path $integrationRoot 'engine-version.json') | ConvertFrom-Json
if ($version.MajorVersion -ne $expected.major -or $version.MinorVersion -ne $expected.minor -or
    $version.PatchVersion -ne $expected.patch -or $version.Changelist -ne $expected.changelist) {
    throw 'Engine build differs from integrations/unreal/engine-version.json. Validate the new build explicitly.'
}

$cacheRoot = Join-Path $projectRoot '.cache\unreal'
New-Item -ItemType Directory -Force $cacheRoot | Out-Null
[Environment]::SetEnvironmentVariable('UE-LocalDataCachePath', (Join-Path $cacheRoot 'DerivedDataCache'), 'Process')
$commonArgs = @($projectFile, "-UserDir=$cacheRoot\User", '-NoSplash', '-NoLiveCoding', '-NoP4', '-ddc=InstalledNoZenLocalFallback', '-NoZen', '-corelimit=2')
$commonArgs += "-ShaderWorkingDir=$cacheRoot\ShaderWorking"
if ($Action -eq 'Open') {
    $editor = Join-Path $EngineRoot 'Engine\Binaries\Win64\UnrealEditor.exe'
    # A visible window is intentional: this action opens the user's editor.
    $boneMap = Join-Path $integrationRoot 'MyoD1\Content\Generated\Bones_v1\L_Bones_90.umap'
    if (Test-Path -LiteralPath $boneMap) {
        $commonArgs = @($projectFile, '/Game/Generated/Bones_v1/L_Bones_90') + $commonArgs[1..($commonArgs.Count - 1)]
    }
    Start-Process -FilePath $editor -ArgumentList ($commonArgs | ForEach-Object { '"' + $_ + '"' })
    exit 0
}

$editor = Join-Path $EngineRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$scriptFile = Join-Path $integrationRoot 'scripts\probe_editor.py'
$reportFile = Join-Path $cacheRoot 'User\Saved\Myo\editor-probe.json'
if ($Action -eq 'FleshProbe') {
    $commonArgs += '-myo-flesh-probe'
    $reportFile = Join-Path $cacheRoot 'User\Saved\Myo\flesh-probe.json'
}
if ($Action -eq 'PrepareBones') {
    Push-Location $projectRoot
    try {
        & (Join-Path $projectRoot '.venv\Scripts\python.exe') -m myo.unreal_geometry
        if ($LASTEXITCODE -ne 0) { throw 'MyoArm geometry export failed.' }
    } finally { Pop-Location }
    $scriptFile = Join-Path $integrationRoot 'scripts\prepare_bones.py'
    $reportFile = Join-Path $cacheRoot 'User\Saved\Myo\bones-import.json'
}
if ($Action -eq 'Capture') {
    $scriptFile = Join-Path $integrationRoot 'scripts\capture_bones.py'
    $reportFile = Join-Path $cacheRoot 'User\Saved\Myo\bones-capture.json'
}
if (Test-Path -LiteralPath $reportFile) { Remove-Item -LiteralPath $reportFile }
if ($Action -eq 'Capture') {
    & $editor @commonArgs '-unattended' '-NoSound' '-RenderOffscreen' '-windowed' '-ResX=1280' '-ResY=720' "-ExecutePythonScript=$scriptFile" '-stdout' '-FullStdOutLogOutput'
} else {
    & $editor @commonArgs '-unattended' '-NullRHI' '-NoSound' '-run=pythonscript' "-script=$scriptFile" '-stdout' '-FullStdOutLogOutput'
}
$editorExitCode = $LASTEXITCODE
if (-not (Test-Path -LiteralPath $reportFile)) { throw 'Editor did not write its probe report.' }
$report = Get-Content -Raw $reportFile | ConvertFrom-Json
$report | Add-Member -NotePropertyName python_script_ok -NotePropertyValue $report.ok
$report | Add-Member -NotePropertyName editor_exit_code -NotePropertyValue $editorExitCode
$report.ok = $report.ok -and ($editorExitCode -eq 0)
$report | ConvertTo-Json -Depth 20 | Set-Content -Encoding UTF8 $reportFile
if ($editorExitCode -ne 0) { throw "Unreal $Action failed: exit $editorExitCode. See $reportFile" }
if (-not $report.ok) { throw "Editor probe reported failure. See $reportFile" }
Write-Output "$Action passed. Report: $reportFile"
