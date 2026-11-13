# Supplementary check (run after the main campaign, on AC power): effective CPU clock while the
# blocked kernel runs with p threads. Samples the Windows counter "% Processor Performance"
# (percent of the 2.0 GHz base clock) every 2 s while build/parallel.exe runs, averaged over the
# logical processors the threads are pinned to (spread order) and over all 16 (_Total).
#
#   powershell -ExecutionPolicy Bypass -File scripts/frequency_check.ps1 -Data $HOME/mm3_work/data
param([string]$Data = "$HOME/mm3_work/data", [int]$N = 4096, [int[]]$Threads = @(1, 2, 4, 8, 16),
      [string]$Out = "results/frequency_check.csv")
[System.Threading.Thread]::CurrentThread.CurrentCulture = [Globalization.CultureInfo]::InvariantCulture
$exe = Resolve-Path "build/parallel.exe"
$rows = @()
foreach ($p in $Threads) {
    $env:OMP_NUM_THREADS = "$p"; $env:OMP_DYNAMIC = "FALSE"
    $log = New-TemporaryFile
    $args_ = "bench --method blocked_omp --n $N --a $Data/A_$N.bin --b $Data/B_$N.bin " +
             "--threads $p --affinity spread --warmup 1 --reps 3"
    $proc = Start-Process -FilePath $exe -ArgumentList $args_ -NoNewWindow -PassThru `
            -RedirectStandardOutput $log.FullName
    $spread = @(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15)
    $active = @(0..([math]::Min($p, 16) - 1) | ForEach-Object { "0,$($spread[$_])" })
    $samples = @(); $totals = @()
    Start-Sleep -Seconds 3          # skip input loading
    while (-not $proc.HasExited) {
        $all = Get-CimInstance Win32_PerfFormattedData_Counters_ProcessorInformation
        $samples += ($all | Where-Object { $active -contains $_.Name } |
                     Measure-Object -Property PercentProcessorPerformance -Average).Average
        $totals += [double]($all | Where-Object { $_.Name -eq '_Total' }).PercentProcessorPerformance
        Start-Sleep -Seconds 2
    }
    $times = Get-Content $log.FullName | Where-Object { $_ -match ',measure,' } |
             ForEach-Object { [double]($_.Split(',')[7]) } | Sort-Object
    $med = $times[[int][math]::Floor($times.Count / 2)]
    $perf = ($samples | Measure-Object -Average).Average
    $tot = ($totals | Measure-Object -Average).Average
    $rows += [pscustomobject]@{
        method = "blocked_omp"; n = $N; threads = $p; samples = $samples.Count
        pct_perf_active_lp = [math]::Round($perf, 1)
        clock_active_lp_ghz = [math]::Round(2.0 * $perf / 100, 2)
        pct_perf_total = [math]::Round($tot, 1)
        median_time_s = $med; gflops = [math]::Round(2 * [math]::Pow($N, 3) / $med / 1e9, 2)
    }
    Remove-Item $log.FullName
    Start-Sleep -Seconds 5
}
$rows | Export-Csv -NoTypeInformation -Encoding utf8 $Out
$rows | Format-Table -AutoSize
