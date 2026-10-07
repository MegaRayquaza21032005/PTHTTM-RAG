$ErrorActionPreference = "Stop"

function Assert-True {
    param(
        [bool]$Condition,
        [string]$Message
    )

    if (-not $Condition) {
        throw "FAIL: $Message"
    }
}

$frontendRoot = Split-Path -Parent $PSScriptRoot
$indexPath = Join-Path $frontendRoot "index.html"
$stylesPath = Join-Path $frontendRoot "styles.css"
$appPath = Join-Path $frontendRoot "app.js"
$dockerfilePath = Join-Path $frontendRoot "Dockerfile"

foreach ($path in @($indexPath, $stylesPath, $appPath, $dockerfilePath)) {
    Assert-True (Test-Path $path) "Missing required file: $path"
}

$index = Get-Content -Raw -Encoding UTF8 $indexPath
$styles = Get-Content -Raw -Encoding UTF8 $stylesPath
$app = Get-Content -Raw -Encoding UTF8 $appPath
$dockerfile = Get-Content -Raw -Encoding UTF8 $dockerfilePath

Assert-True ($index -match 'id="question-form"') "Question form is missing"
Assert-True ($index -match 'id="question-input"') "Question input is missing"
Assert-True ($index -match 'id="messages"') "Accessible message region is missing"
Assert-True ($index -match 'id="connection-status"') "Backend status indicator is missing"
Assert-True ($index -match 'meta name="viewport"') "Responsive viewport metadata is missing"
Assert-True ($index -match 'styles\.css\?v=score-v2') "Stylesheet cache-busting version is missing"
Assert-True ($index -match 'app\.js\?v=score-v2') "JavaScript cache-busting version is missing"
Assert-True ($index -match 'id="ui-version"') "Visible UI build marker is missing"

Assert-True ($app -match 'http://localhost:8001/api') "Frontend does not target the local chatbot API"
Assert-True ($app -match '/health') "Frontend does not check backend health"
Assert-True ($app -match '/chat') "Frontend does not call the chat endpoint"
Assert-True ($app -match 'method:\s*["'']POST["'']') "Chat request is not a POST"
Assert-True ($app -match 'application/json') "Chat request does not send JSON"
Assert-True ($app -match 'renderSources') "Source rendering is missing"
Assert-True ($app -match 'renderError') "Visible error handling is missing"
Assert-True ($app -match 'textContent') "Dynamic content must be rendered as text"
Assert-True ($app -match '\.normalize\(["'']NFC["'']\)') "API text is not normalized to Unicode NFC"
Assert-True ($app -match 'source-index') "Numbered source cards are missing"
Assert-True ($app -match 'source-identity') "Source title and document are not grouped"
Assert-True ($app -match 'source-document') "Source document label is missing"
Assert-True ($app -match 'score_type') "Frontend does not distinguish RRF and rerank scores"
Assert-True ($app -match '\u0110i\u1EC3m RRF') "RRF score label is missing"
Assert-True ($app -match '\u0110i\u1EC3m rerank') "Rerank score label is missing"
Assert-True ($app -match 'toFixed\(4\)') "Raw scores are not displayed with stable precision"
Assert-True ($app -notmatch 'source\.score\s*\*\s*100') "Score must not be converted to a percentage"
Assert-True ($app -match 'source-score-note') "Visible score explanation is missing"
Assert-True ($app -match 'kh\u00F4ng ph\u1EA3i %') "Score explanation must say it is not a percentage"
Assert-True ($app -match 'RRF:') "RRF score type is not explicit"
Assert-True ($app -match 'Rerank:') "Rerank score type is not explicit"

Assert-True ($styles -match '@media') "Responsive styles are missing"
Assert-True ($styles -match ':focus-visible') "Keyboard focus styles are missing"
Assert-True ($styles -match 'prefers-reduced-motion') "Reduced-motion support is missing"
Assert-True ($styles -match '\.source-identity') "Source identity layout is missing"
Assert-True ($styles -match 'overflow-wrap:\s*anywhere') "Long source text is not protected from overflow"
Assert-True ($styles -match '--serif:\s*"Times New Roman"') "Primary serif font must support Vietnamese composition on Windows"
Assert-True ($styles -match '\.source-score-note') "Score explanation styling is missing"

Assert-True ($dockerfile -match 'EXPOSE\s+5173') "Frontend container does not expose port 5173"
Assert-True ($dockerfile -match 'http\.server["'',\s]+5173') "Frontend container does not serve on port 5173"

Write-Output "PASS: frontend acceptance checks"
