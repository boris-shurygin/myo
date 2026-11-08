param(
    [ValidateSet('bones', 'muscles')]
    [string]$View = 'bones',
    [ValidateSet(15, 45, 90, 120)]
    [int]$Pose = 90,
    [ValidateRange(15, 120)]
    [int]$Seconds = 120
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python3 = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python3)) {
    throw 'Run scripts/setup-windows.ps1 first.'
}
$viewName = if ($View -eq 'muscles') { 'Muscles' } else { 'Bones' }
$mapFile = Join-Path $projectRoot "integrations\unreal\MyoD1\Content\Generated\${viewName}_v1\L_${viewName}_$Pose.umap"
if (-not (Test-Path -LiteralPath $mapFile)) {
    throw 'Prepare the local MyoArm scene with the prepare-bones profiler stage first.'
}
Write-Host 'Camera: mouse to look; W/A/S/D to move; Q/E or Ctrl/Space down/up.'
Write-Host 'Home: reset view. Esc: close. Blue ball at the hand: model payload.'
if ($View -eq 'muscles') {
    Write-Host 'BodyParts3D, (c) The Database Center for Life Science licensed under CC Attribution 4.0 International.'
    Write-Host 'Preliminary pose surfaces; active muscle deformation is not implemented yet.'
}
& $python3 (Join-Path $PSScriptRoot 'profile_unreal.py') --stage viewer --visible --seconds $Seconds --fps 20 --pose $Pose --view $View
exit $LASTEXITCODE
