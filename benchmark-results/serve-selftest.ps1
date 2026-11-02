# serve-selftest.ps1 - the coding-assistant path end to end: turbo-qwen36.bat serve (the
# OpenAI-compatible API), three chat turns, timings, then the server stopped. Exit 0 = all
# turns answered. Run: powershell -ExecutionPolicy Bypass -File benchmark-results\serve-selftest.ps1
param([int]$Port = 8000, [int]$StartTimeoutSec = 600)
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$log = Join-Path $here 'serve-selftest.log'
$base = "http://127.0.0.1:$Port"

$proc = Start-Process -FilePath cmd.exe -ArgumentList '/c', "`"$here\turbo-qwen36.bat`" --port $Port > `"$log`" 2>&1" `
    -WindowStyle Hidden -PassThru
function Stop-Server { if ($proc -and -not $proc.HasExited) { & taskkill /PID $proc.Id /T /F | Out-Null } }

try {
    $t0 = Get-Date
    while ($true) {
        if ($proc.HasExited) { throw "the server exited during start; see $log" }
        try { if ((Invoke-WebRequest -UseBasicParsing "$base/health" -TimeoutSec 5).StatusCode -eq 200) { break } } catch { }
        if (((Get-Date) - $t0).TotalSeconds -gt $StartTimeoutSec) { throw "no /health 200 within $StartTimeoutSec s; see $log" }
        Start-Sleep -Seconds 2
    }
    Write-Host ("server up in {0:N0} s" -f ((Get-Date) - $t0).TotalSeconds)
    $model = (Invoke-RestMethod "$base/v1/models").data[0].id

    $code = @'
def merge(a, b):
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    return out
'@
    $history = @(@{ role = 'system'; content = 'You are a concise coding assistant.' })
    $turns = @(
        'Write a Python function is_palindrome(s) that ignores case and punctuation. Code only.',
        "This merge function has a bug. Fix it and show the corrected code only:`n$code",
        'Now add type hints to the fixed function. Code only.'
    )
    $failed = 0
    foreach ($q in $turns) {
        $history += @{ role = 'user'; content = $q }
        $body = @{ model = $model; messages = $history; max_tokens = 256; temperature = 0; enable_thinking = $false } |
            ConvertTo-Json -Depth 6
        $s = Get-Date
        try {
            $r = Invoke-RestMethod -Method Post "$base/v1/chat/completions" -ContentType 'application/json' -Body $body -TimeoutSec 900
            $dt = ((Get-Date) - $s).TotalSeconds
            $ans = $r.choices[0].message.content
            $n = $r.usage.completion_tokens
            Write-Host ("turn {0}: {1} tokens in {2:N1} s ({3:N2} tok/s end to end, prompt {4} tokens)" -f `
                ($history.Count / 2), $n, $dt, ($n / [math]::Max($dt, 0.001)), $r.usage.prompt_tokens)
            Write-Host ($ans.Substring(0, [math]::Min(300, $ans.Length)))
            Write-Host '---'
            $history += @{ role = 'assistant'; content = $ans }
        } catch {
            $failed++
            Write-Host "turn failed: $($_.Exception.Message)"
        }
    }
    if ($failed) { Write-Host "$failed turn(s) failed; see $log"; exit 1 }
    Write-Host "all turns answered; server log: $log"
    exit 0
} finally {
    Stop-Server
}
