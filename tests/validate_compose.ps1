$ErrorActionPreference = "Stop"

function Assert-Match {
    param(
        [string]$Content,
        [string]$Pattern,
        [string]$Message
    )

    if ($Content -notmatch $Pattern) {
        throw "FAIL: $Message"
    }
}

$root = Split-Path -Parent $PSScriptRoot
$composePath = Join-Path $root "compose.yaml"
$readmePath = Join-Path $root "README.md"
$devServerPath = Join-Path $root "frontend/dev_server.py"

if (-not (Test-Path $composePath)) {
    throw "FAIL: compose.yaml is missing"
}

if (-not (Test-Path $devServerPath)) {
    throw "FAIL: frontend/dev_server.py is missing"
}

$compose = Get-Content -Raw -Encoding UTF8 $composePath
$readme = Get-Content -Raw -Encoding UTF8 $readmePath
$devServer = Get-Content -Raw -Encoding UTF8 $devServerPath

Assert-Match $compose '(?m)^\s{2}chatbot:\s*$' "chatbot service is missing"
Assert-Match $compose '(?m)^\s{2}frontend:\s*$' "frontend service is missing"
Assert-Match $compose 'chatbot/app:/app/app' "chatbot source bind mount is missing"
Assert-Match $compose 'frontend:/app' "frontend source bind mount is missing"
Assert-Match $compose '--reload' "Uvicorn hot reload is not enabled"
Assert-Match $compose 'WATCHFILES_FORCE_POLLING' "Docker Desktop polling is not enabled"
Assert-Match $compose 'dev_server\.py' "frontend live-reload server is not configured"
Assert-Match $compose '8001:8001' "chatbot port is missing"
Assert-Match $compose '5173:5173' "frontend port is missing"
Assert-Match $compose 'chatbot/\.env' "chatbot env file is not configured"

Assert-Match $devServer 'text/event-stream' "frontend server does not expose an SSE stream"
Assert-Match $devServer 'EventSource' "frontend server does not inject the reload client"
Assert-Match $devServer 'location\.reload' "frontend server does not reload the browser"

Assert-Match $readme 'docker compose up' "README does not document startup"
Assert-Match $readme 'docker compose down' "README does not document shutdown"
Assert-Match $readme 'hot reload' "README does not explain hot reload"
Assert-Match $readme 'http://localhost:5173' "README does not link the frontend"
Assert-Match $readme 'http://localhost:8001/docs' "README does not link API docs"

Write-Output "PASS: Docker Compose development workflow"
