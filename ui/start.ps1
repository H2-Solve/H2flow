# Starts everything the H2 Solve UI needs and opens the page.
# Double-click ui\start.bat, or run:  powershell -ExecutionPolicy Bypass -File ui\start.ps1
# Press Ctrl+C in this window to stop the connector.

$root = Split-Path $PSScriptRoot -Parent
$page = Join-Path $PSScriptRoot 'H2Solve_v2_0.html'
$venvPython = Join-Path $root '.venv\Scripts\python.exe'
$model = 'gemma3:4b'
$port = 5001
Set-Location $root

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`n$text" -ForegroundColor Red; Read-Host 'Press Enter to close'; exit 1 }

function Test-Url($url) {
    try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null; return $true } catch { return $false }
}

Step 'Ollama'
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Fail 'Ollama is not installed. Get it from https://ollama.com/download and run this again.'
}
if (-not (Test-Url 'http://127.0.0.1:11434/api/version')) {
    Write-Host 'Starting Ollama...'
    $app = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama app.exe'
    if (Test-Path $app) { Start-Process $app } else { Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden }
    $deadline = (Get-Date).AddSeconds(30)
    while (-not (Test-Url 'http://127.0.0.1:11434/api/version')) {
        if ((Get-Date) -gt $deadline) { Fail 'Ollama did not start. Open it from the Start menu and run this again.' }
        Start-Sleep -Seconds 1
    }
}
if (-not ((ollama list) -match [regex]::Escape($model))) {
    Write-Host "Downloading $model (about 3 GB, first time only)..."
    ollama pull $model
    if ($LASTEXITCODE -ne 0) { Fail "Could not download $model." }
}
Write-Host 'Ollama is running.'

Step 'Python environment'
if (-not (Test-Path $venvPython)) {
    Write-Host 'Creating .venv (first time only)...'
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { Fail 'Could not create .venv. Is Python 3.10+ installed?' }
}
& $venvPython -c 'import fastapi, uvicorn, elasticsearch, langchain_ollama, sympy' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing packages (first time or after requirements change)...'
    & $venvPython -m pip install -q -r requirements.txt -r ui\requirements.txt
    if ($LASTEXITCODE -ne 0) { Fail 'Package install failed; see the messages above.' }
}
Write-Host 'Packages are installed.'

# Optional: the RAG card needs Elasticsearch (Docker) and an indexed textbook.
Step 'Elasticsearch (for the RAG card)'
if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Could not start Elasticsearch. Is Docker Desktop open? The RAG card will show an error.' -ForegroundColor Yellow
    } elseif (-not (Test-Url 'http://localhost:9201/textbook-chunks/_count')) {
        Write-Host 'Elasticsearch may still be starting, or the textbook is not indexed yet. To index it:' -ForegroundColor Yellow
        Write-Host '  git clone https://github.com/openstax/osbooks-calculus-bundle ..\osbooks-calculus-bundle'
        Write-Host '  .venv\Scripts\python -m search.index --bundle ..\osbooks-calculus-bundle --book calculus-volume-1 --reset'
    } else {
        Write-Host 'Elasticsearch is running with the textbook indexed.'
    }
} else {
    Write-Host 'Docker is not installed, so the RAG card will show an error. The Default card still works.' -ForegroundColor Yellow
}

Step 'Connector'
if (Test-Url "http://127.0.0.1:$port/api/model") {
    Write-Host "A connector is already running on port $port. Opening the page."
    Start-Process $page
    exit 0
}

# Open the page once the connector answers, so the status light sees it on load.
Start-Job -ArgumentList $port, $page -ScriptBlock {
    param($port, $page)
    for ($i = 0; $i -lt 60; $i++) {
        try { Invoke-WebRequest "http://127.0.0.1:$port/api/model" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop | Out-Null; break }
        catch { Start-Sleep -Seconds 1 }
    }
    Start-Process $page
} | Out-Null

Write-Host "Starting on http://127.0.0.1:$port. The page opens when it is ready. Press Ctrl+C to stop."
& $venvPython -m uvicorn ui.connector:app --port $port
