# Builds dist\Neurox.exe (Windows, single file).
# Usage (from the neurox-LLM folder):  .\build_exe.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    python -m venv .venv
}
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt pyinstaller

Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
.\.venv\Scripts\python -m PyInstaller --noconfirm --clean neurox.spec

Write-Host ""
Write-Host "Done: $PSScriptRoot\dist\Neurox.exe"
