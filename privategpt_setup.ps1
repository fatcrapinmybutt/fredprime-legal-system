param(
    [string]$InstallPath = "C:\\privategpt",
    [string]$RepoUrl = "https://github.com/zylon-ai/private-gpt.git",
    [string]$Model = "mistral"
)

if (!(Test-Path $InstallPath)) {
    New-Item -ItemType Directory -Path $InstallPath | Out-Null
}

if (!(Test-Path "$InstallPath\.git")) {
    git clone $RepoUrl $InstallPath
}

Set-Location $InstallPath

if (!(Test-Path "$InstallPath\venv")) {
    python -m venv venv
}

& "$InstallPath\venv\Scripts\Activate.ps1"

if (-not (Get-Command poetry -ErrorAction SilentlyContinue)) {
    pip install poetry
}

poetry install --extras ui
pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m private_gpt --config settings-local.yaml
