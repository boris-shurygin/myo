param([string]$Uv = 'uv')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
$previousCache = $env:UV_CACHE_DIR
$previousPythonDir = $env:UV_PYTHON_INSTALL_DIR
try {
    $env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $projectRoot '.cache\python'
    $python3 = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath '.venv') {
        if (-not (Test-Path -LiteralPath $python3)) {
            throw 'Existing .venv is not a Windows environment. Preserve it before running setup.'
        }
        & $python3 -I -c 'import sys; sys.exit(not (sys.version_info[:2] == (3, 12) and sys.prefix != sys.base_prefix))'
        if ($LASTEXITCODE -ne 0) { throw 'Existing .venv is incompatible; Python 3.12 is required.' }
    } else {
        & $Uv venv --managed-python --python 3.12 --seed .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create Python 3.12 environment.' }
    }
    & $Uv pip install --python $python3 --editable '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    & $python3 -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependency validation failed.' }
    & $python3 -I -c "import sys, myo, mujoco, numpy, gymnasium; print('Ready:', sys.executable, sys.version.split()[0]); print('MuJoCo', mujoco.__version__, 'NumPy', numpy.__version__, 'Gymnasium', gymnasium.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'Project import failed.' }
} finally {
    $env:UV_CACHE_DIR = $previousCache
    $env:UV_PYTHON_INSTALL_DIR = $previousPythonDir
    Pop-Location
}
