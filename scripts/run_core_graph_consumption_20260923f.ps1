$ErrorActionPreference = 'Stop'
$repo = 'D:\python_projects\zero_ai\pure_integer_ai'; $source = Join-Path $repo 'src'
$database = 'K:\pure_integer_ai_work\current_runs\discourse-relation-response-graph-20260917b\training.sqlite3'
$inventoryRoot = 'K:\pure_integer_ai_work\current_runs\core-graph-input-inventory-20260923e'
$heldoutRoot = 'K:\pure_integer_ai_work\current_runs\core-graph-heldout-20260920r5'
$runRoot = 'K:\pure_integer_ai_work\current_runs\core-graph-consumption-20260923f'
$control = 'K:\pure_integer_ai_work\orchestration\core-graph-consumption-20260923f-control'
$state = Join-Path $control 'state.json'; $stdout = Join-Path $control 'stdout.txt'; $stderr = Join-Path $control 'stderr.txt'
function Save-State([hashtable] $value) { ($value | ConvertTo-Json -Depth 12 -Compress) | Set-Content -LiteralPath $state -Encoding ascii }
function Integer-Surface([int[]] $values) { return -join ($values | ForEach-Object { [char] $_ }) }
New-Item -ItemType Directory -Force -Path $control | Out-Null
try {
  if (Test-Path -LiteralPath $runRoot) { throw "run root already exists: $runRoot" }
  $inventory = Get-Content -Raw (Join-Path $inventoryRoot 'core_graph_input_keys.int.json') | ConvertFrom-Json
  $keys = @($inventory[2] | ForEach-Object { if ($_[1].Count -ne 1) { throw 'inventory kind cardinality invalid' }; (($_[1][0] | ForEach-Object { [string]$_ }) -join ',') })
  if ($inventory[0] -ne 91580 -or $inventory[1] -ne 1 -or $keys.Count -ne 3) { throw 'inventory payload invalid' }
  $course = Join-Path $heldoutRoot 'heldout_response_course.int.json'; $manifest = Join-Path $heldoutRoot 'heldout_manifest.json'
  $input1 = Integer-Surface @(25105,24819,21040,36824,19981,21547,36710,30340,32467,20837,12290)
  $input2 = Integer-Surface @(36827,28982,21040,21364,19968,20010,26032,36824,21487,20197,30340,21547,36710,65311)
  $input3 = Integer-Surface @(36825,20010,32467,30340,19968,20010,26032,30340,65292,25105,24819,23450,20294,30340,26590,20040,12290)
  $env:PYTHONPATH = $source
  Save-State @{format='PURE_INTEGER_CORE_GRAPH_CONSUMPTION_CONTROL_V4';status='RUNNING';started_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;inventory_root=$inventoryRoot;free_dialogue_complete=0}
  $process = Start-Process -FilePath 'python' -ArgumentList @(
    '-u','-m','pure_integer_ai.experiments.run_unknown_input_structure_slice',
    '--database',$database,'--run-root',$runRoot,'--session-id','92110',
    '--input',$input1,'--input',$input2,'--input',$input3,
    '--graph-object-key',$keys[0],'--graph-object-key',$keys[1],'--graph-object-key',$keys[2],
    '--heldout-course',$course,'--heldout-manifest',$manifest
  ) -WorkingDirectory $repo -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  $process.WaitForExit(); $exit = $process.ExitCode
  if ($exit -ne 0) { throw "core graph consumption exit $exit" }
  $receipt = Join-Path $runRoot 'receipt.json'; if (-not (Test-Path -LiteralPath $receipt)) { throw 'receipt missing' }
  $value = Get-Content -Raw $receipt | ConvertFrom-Json
  Save-State @{format='PURE_INTEGER_CORE_GRAPH_CONSUMPTION_CONTROL_V4';status='PASS';completed_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;receipt=$receipt;graph_input_count=[int]$value.graph_input_count;core_graph_input_consumed_count=[int]$value.core_graph_input_consumed_count;generation_consumed_graph_input_count=[int]$value.generation_consumed_graph_input_count;heldout_match_count=[int]$value.heldout_match_count;delivery_count=[int]$value.delivery_count;free_dialogue_complete=0}
} catch { $_ | Out-String | Set-Content -LiteralPath $stderr -Encoding utf8; Save-State @{format='PURE_INTEGER_CORE_GRAPH_CONSUMPTION_CONTROL_V4';status='FAIL';completed_utc=[DateTime]::UtcNow.ToString('o');run_root=$runRoot;error=$_.Exception.Message;free_dialogue_complete=0}; exit 1 }
