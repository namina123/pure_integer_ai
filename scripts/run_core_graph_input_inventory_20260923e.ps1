$ErrorActionPreference = 'Stop'
$repo = 'D:\python_projects\zero_ai\pure_integer_ai'; $source = Join-Path $repo 'src'
$database = 'K:\pure_integer_ai_work\current_runs\discourse-relation-response-graph-20260917b\training.sqlite3'
$runRoot = 'K:\pure_integer_ai_work\current_runs\core-graph-input-inventory-20260923e'
$control = 'K:\pure_integer_ai_work\orchestration\core-graph-input-inventory-20260923e-control'
$state = Join-Path $control 'state.json'; $stdout = Join-Path $control 'stdout.txt'; $stderr = Join-Path $control 'stderr.txt'
New-Item -ItemType Directory -Force -Path $control | Out-Null
function Save-State([hashtable] $value) { ($value | ConvertTo-Json -Depth 12 -Compress) | Set-Content -LiteralPath $state -Encoding ascii }
try {
  if (Test-Path -LiteralPath $runRoot) { throw "run root already exists: $runRoot" }
  $env:PYTHONPATH = $source
  $process = Start-Process -FilePath 'python' -ArgumentList @(
    '-u', 'scripts\select_core_graph_input_keys.py', '--database', $database,
    '--run-root', $runRoot, '--per-kind', '1'
  ) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  Save-State @{format='PURE_INTEGER_CORE_GRAPH_INPUT_INVENTORY_CONTROL_V1';status='RUNNING';pid=$process.Id;started_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;free_dialogue_complete=0}
  Wait-Process -Id $process.Id; $process.Refresh()
  if ($process.ExitCode -ne 0) { throw "inventory exit $($process.ExitCode)" }
  $receipt = Get-Content -Raw -LiteralPath (Join-Path $runRoot 'receipt.json') | ConvertFrom-Json
  if ($receipt.object_count -ne 3 -or $receipt.object_kind_count -ne 3 -or $receipt.integer_payload -ne 1 -or $receipt.free_dialogue_complete -ne 0) { throw 'inventory receipt did not close' }
  Save-State @{format='PURE_INTEGER_CORE_GRAPH_INPUT_INVENTORY_CONTROL_V1';status='PASS';completed_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;receipt=(Join-Path $runRoot 'receipt.json');object_count=[int]$receipt.object_count;free_dialogue_complete=0}
} catch { $_ | Out-String | Set-Content -LiteralPath $stderr -Encoding utf8; Save-State @{format='PURE_INTEGER_CORE_GRAPH_INPUT_INVENTORY_CONTROL_V1';status='FAIL';completed_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;error=$_.Exception.Message;free_dialogue_complete=0}; exit 1 }
