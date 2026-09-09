# Launch Ollama/Mistral first, then Chatterbox and RVC.
# The order is intentional: on this 8GB GPU, loading the voice models first can
# make Ollama pin Mistral to CPU (observed 14-15s replies instead of 1-3s).

$ErrorActionPreference = "Continue"
$root = "d:\voice-assistantv2\voice-assistant"
$ollamaApp = "C:\Users\shku0\AppData\Local\Programs\Ollama\ollama app.exe"

function Test-LocalPort([int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        return $task.Wait(400) -and $client.Connected
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Wait-LocalPort([int]$Port, [int]$TimeoutSeconds) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        if (Test-LocalPort $Port) { return $true }
        Start-Sleep -Milliseconds 400
    } while ((Get-Date) -lt $deadline)
    return $false
}

Write-Host "=== Starting assistant services ===" -ForegroundColor Cyan

# 1. Ollama service.
if (-not (Test-LocalPort 11434)) {
    Write-Host "[1/4] Starting Ollama..." -ForegroundColor Yellow
    Start-Process $ollamaApp
    if (-not (Wait-LocalPort 11434 30)) {
        Write-Host "Ollama did not open port 11434." -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[1/4] Ollama already running." -ForegroundColor DarkYellow
}

# 2. Warm Mistral before either voice model reserves VRAM. Only rebalance when
# both voice services are down; unloading a live setup would interrupt speech.
$ttsWasUp = Test-LocalPort 7866
$rvcWasUp = Test-LocalPort 7865
if (-not $ttsWasUp -and -not $rvcWasUp) {
    Write-Host "[2/4] Loading Mistral into GPU memory..." -ForegroundColor Yellow
    & ollama stop mistral 2>$null | Out-Null

    $warmBody = @{
        model = "mistral"
        messages = @(@{ role = "user"; content = "hi" })
        stream = $false
        keep_alive = "30m"
        options = @{
            num_predict = 1
            temperature = 0.8
            top_p = 0.9
            num_ctx = 2048
            num_thread = 12
        }
    } | ConvertTo-Json -Depth 6

    try {
        Invoke-RestMethod `
            -Uri "http://127.0.0.1:11434/api/chat" `
            -Method Post `
            -ContentType "application/json" `
            -Body $warmBody `
            -TimeoutSec 180 | Out-Null
        Write-Host "      Mistral is warm." -ForegroundColor Green
    } catch {
        Write-Host "      Mistral warmup failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "[2/4] Voice service already active; leaving loaded models untouched." -ForegroundColor DarkYellow
}

# 3. Chatterbox TTS.
if (-not $ttsWasUp) {
    Write-Host "[3/4] Starting Chatterbox TTS..." -ForegroundColor Yellow
    Start-Process -FilePath "D:\rvc\tts_env\Scripts\python.exe" `
        -ArgumentList "tts_server.py" `
        -WorkingDirectory $root `
        -WindowStyle Normal
} else {
    Write-Host "[3/4] Chatterbox TTS already running." -ForegroundColor DarkYellow
}

# 4. RVC. It can load alongside TTS after Mistral has claimed its GPU layers.
if (-not $rvcWasUp) {
    Write-Host "[4/4] Starting RVC..." -ForegroundColor Yellow
    Start-Process -FilePath "D:\rvc\rvc_env\Scripts\python.exe" `
        -ArgumentList "rvc_server.py" `
        -WorkingDirectory $root `
        -WindowStyle Normal
} else {
    Write-Host "[4/4] RVC already running." -ForegroundColor DarkYellow
}

Write-Host ""
Write-Host "=== Services launched ===" -ForegroundColor Green
Write-Host "Wait for ports 7865 and 7866 before sending the first message."
