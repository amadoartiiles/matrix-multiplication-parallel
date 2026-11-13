# Keeps Windows awake (no idle sleep or Modern Standby) until the file given as the first
# argument exists. Started and stopped by scripts/run_campaign.sh.
param([string]$StopFile = (Join-Path $PSScriptRoot 'stop_keep_awake'))
Add-Type -Namespace W -Name P -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
while (-not (Test-Path $StopFile)) {
    [W.P]::SetThreadExecutionState([uint32]"0x80000003") | Out-Null   # CONTINUOUS | SYSTEM | DISPLAY
    Start-Sleep -Seconds 30
}
[W.P]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null
