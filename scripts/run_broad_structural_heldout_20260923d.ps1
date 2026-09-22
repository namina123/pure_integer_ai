$ErrorActionPreference = 'Stop'
$repo = 'D:\python_projects\zero_ai\pure_integer_ai'; $source = Join-Path $repo 'src'
$database = 'K:\pure_integer_ai_work\current_runs\discourse-relation-response-graph-20260917b\training.sqlite3'; $courseRoot = 'K:\pure_integer_ai_work\current_runs\broad-structural-heldout-course-20260923d'; $runRoot = 'K:\pure_integer_ai_work\current_runs\broad-structural-heldout-session-20260923d'; $control = 'K:\pure_integer_ai_work\orchestration\broad-structural-heldout-20260923d-control'; $state = Join-Path $control 'state.json'; $stdout = Join-Path $control 'session.stdout.txt'; $stderr = Join-Path $control 'session.stderr.txt'
New-Item -ItemType Directory -Force -Path $control | Out-Null
function Save-State([hashtable] $value) { ($value | ConvertTo-Json -Depth 12 -Compress) | Set-Content -LiteralPath $state -Encoding ascii }
try {
  if (Test-Path -LiteralPath $runRoot) { throw "session root already exists: $runRoot" }
  $course = Join-Path $courseRoot 'heldout_response_course.int.json'; $manifest = Join-Path $courseRoot 'heldout_manifest.json'; $env:PYTHONPATH = $source
  function Integer-Surface([int[]] $values) { return -join ($values | ForEach-Object { [char] $_ }) }
  $in1 = Integer-Surface @(27773,36710); $in3 = Integer-Surface @(20027,39064)
  $g1 = '7,0,0,0,1,1,1,1,1,15,1,206,3917972928851022813,0,0,0,0,1,1,1,1,1,2,1,1'; $g3 = '7,0,0,0,1,1,1,1,1,15,1,206,585343985805089038,0,0,0,0,1,1,1,1,1,2,1,1'
  Save-State @{format='PURE_INTEGER_BROAD_STRUCTURAL_HELDOUT_V1';status='RUNNING';started_utc=[DateTime]::UtcNow.ToString('o');session_root=$runRoot;course_root=$courseRoot;free_dialogue_complete=0}
  & python -u -m pure_integer_ai.experiments.run_unknown_input_structure_slice --database $database --run-root $runRoot --session-id 92103 --input $in1 --input $in3 --graph-object-key $g1 --graph-object-key $g3 --heldout-course $course --heldout-manifest $manifest *> $stdout
  if ($LASTEXITCODE -ne 0) { throw "heldout session exit $LASTEXITCODE" }
  $receipt = Join-Path $runRoot 'receipt.json'; if (-not (Test-Path -LiteralPath $receipt)) { throw 'heldout receipt missing' }; $value = Get-Content -Raw $receipt | ConvertFrom-Json
  if ($value.heldout_match_count -ne $value.graph_input_count -or $value.generation_consumed_graph_input_count -ne $value.graph_input_count -or $value.response_delivery_count -ne $value.graph_input_count) { throw 'heldout gates did not close' }
  Save-State @{format='PURE_INTEGER_BROAD_STRUCTURAL_HELDOUT_V1';status='PASS';completed_utc=[DateTime]::UtcNow.ToString('o');session_root=$runRoot;receipt=$receipt;course_root=$courseRoot;free_dialogue_complete=0}
} catch { $_ | Out-String | Set-Content -LiteralPath $stderr -Encoding utf8; Save-State @{format='PURE_INTEGER_BROAD_STRUCTURAL_HELDOUT_V1';status='FAIL';completed_utc=[DateTime]::UtcNow.ToString('o');session_root=$runRoot;course_root=$courseRoot;error=$_.Exception.Message;free_dialogue_complete=0}; exit 1 }
