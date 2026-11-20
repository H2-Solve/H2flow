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

# The RAG card needs Elasticsearch (in Docker) with the textbook indexed.
# Any problem here is a warning only: the Default card works without it.
function Start-Rag {
    $esUrl = 'http://127.0.0.1:9201'
    $docker = 'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
    $dockerApp = 'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    $bundle = Join-Path (Split-Path $root -Parent) 'osbooks-calculus-bundle'

    if (Get-Command docker -ErrorAction SilentlyContinue) { $docker = 'docker' }
    elseif (-not (Test-Path $docker)) { return 'Docker is not installed (https://www.docker.com/products/docker-desktop).' }

    & $docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        if (-not (Test-Path $dockerApp)) { return 'Docker is not running. Open Docker Desktop and run this again.' }
        Write-Host 'Starting Docker Desktop (can take a minute)...'
        Start-Process $dockerApp
        $deadline = (Get-Date).AddMinutes(3)
        do {
            if ((Get-Date) -gt $deadline) { return 'Docker Desktop did not start. Open it, wait for "Engine running", and run this again.' }
            Start-Sleep -Seconds 3
            & $docker info *> $null
        } while ($LASTEXITCODE -ne 0)
    }

    & $docker compose up -d | Out-Host
    if ($LASTEXITCODE -ne 0) { return 'Could not start the Elasticsearch container; see the messages above.' }
    Write-Host 'Waiting for Elasticsearch...'
    $deadline = (Get-Date).AddMinutes(2)
    while (-not (Test-Url "$esUrl/_cluster/health")) {
        if ((Get-Date) -gt $deadline) { return 'Elasticsearch did not come up in time. Run this again in a minute.' }
        Start-Sleep -Seconds 2
    }

    # 127.0.0.1, not localhost: avoids the Windows IPv6 delay. The connector inherits this.
    $env:H2FLOW_ES_URL = $esUrl

    $count = 0
    try { $count = (Invoke-RestMethod "$esUrl/textbook-chunks/_count" -TimeoutSec 5 -ErrorAction Stop).count } catch { }
    if ($count -gt 0) { Write-Host "Elasticsearch is running with $count textbook chunks."; return $null }

    Write-Host 'Setting up the textbook (first time only, a few minutes)...'
    if (-not (Test-Path $bundle)) {
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) { return 'git is needed to download the textbook.' }
        git clone --depth 1 https://github.com/openstax/osbooks-calculus-bundle $bundle | Out-Host
        if ($LASTEXITCODE -ne 0) { return 'Could not download the textbook.' }
    }
    & $venvPython -m search.index --bundle $bundle --book calculus-volume-1 --reset | Out-Host
    if ($LASTEXITCODE -ne 0) { return 'Indexing the textbook failed; see the messages above.' }
    Write-Host 'Textbook indexed.'
    return $null
}

Step 'Elasticsearch (for the RAG card)'
$ragProblem = Start-Rag
if ($ragProblem) { Write-Host "$ragProblem The RAG card will show an error; the Default card still works." -ForegroundColor Yellow }

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
