$ErrorActionPreference = "Stop"
Set-Location "$PSScriptRoot\backend"
if (Test-Path "..\venv\Scripts\Activate.ps1") { . "..\venv\Scripts\Activate.ps1" }
pip install -q -r requirements.txt
Write-Host "API:  http://127.0.0.1:8000   Docs: http://127.0.0.1:8000/docs"
uvicorn app.main:app --host 127.0.0.1 --port 8000
