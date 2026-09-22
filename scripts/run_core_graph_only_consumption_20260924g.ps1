$ErrorActionPreference = 'Stop'
$repo = 'D:\python_projects\zero_ai\pure_integer_ai'; $source = Join-Path $repo 'src'
$runRoot = 'K:\pure_integer_ai_work\current_runs\core-graph-only-consumption-20260924g'
$control = 'K:\pure_integer_ai_work\orchestration\core-graph-only-consumption-20260924g-control'
$state = Join-Path $control 'state.json'; $stdout = Join-Path $control 'stdout.txt'; $stderr = Join-Path $control 'stderr.txt'
New-Item -ItemType Directory -Force -Path $control | Out-Null
function Save-State([hashtable] $value) { ($value | ConvertTo-Json -Depth 12 -Compress) | Set-Content -LiteralPath $state -Encoding ascii }
try {
  if (Test-Path -LiteralPath $runRoot) { throw "run root already exists: $runRoot" }
  $env:PYTHONPATH = $source
  $process = Start-Process -FilePath 'python' -ArgumentList @('-u','scripts\run_core_graph_only_consumption_20260924g.py') -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  Save-State @{format='PURE_INTEGER_CORE_GRAPH_ONLY_CONSUMPTION_CONTROL_V1';status='RUNNING';pid=$process.Id;started_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;free_dialogue_complete=0}
  $process.WaitForExit(); $exit = $process.ExitCode
  if ($exit -ne 0) { throw "graph-only consumption exit $exit" }
  $receipt = Join-Path $runRoot 'receipt.json'; if (-not (Test-Path -LiteralPath $receipt)) { throw 'graph-only receipt missing' }
  $value = Get-Content -Raw $receipt | ConvertFrom-Json
  Save-State @{format='PURE_INTEGER_CORE_GRAPH_ONLY_CONSUMPTION_CONTROL_V1';status='PASS';completed_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;receipt=$receipt;query_count=[int]$value.query_count;generation_count=[int]$value.generation_count;delivery_count=[int]$value.delivery_count;free_dialogue_complete=0}
} catch { $_ | Out-String | Set-Content -LiteralPath $stderr -Encoding utf8; Save-State @{format='PURE_INTEGER_CORE_GRAPH_ONLY_CONSUMPTION_CONTROL_V1';status='FAIL';completed_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;error=$_.Exception.Message;free_dialogue_complete=0}; exit 1 }
