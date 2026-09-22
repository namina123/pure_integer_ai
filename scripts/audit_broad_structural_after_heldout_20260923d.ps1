$ErrorActionPreference = 'Stop'
$repo = 'D:\python_projects\zero_ai\pure_integer_ai'; $source = Join-Path $repo 'src'
$database = 'K:\pure_integer_ai_work\current_runs\discourse-relation-response-graph-20260917b\training.sqlite3'; $heldoutRoot = 'K:\pure_integer_ai_work\current_runs\broad-structural-heldout-session-20260923d'; $auditRoot = 'K:\pure_integer_ai_work\audits\broad-structural-qualification-20260923d'; $output = Join-Path $auditRoot 'qualification.json'; $control = 'K:\pure_integer_ai_work\orchestration\broad-structural-qualification-20260923d-control'; $state = Join-Path $control 'state.json'; $stdout = Join-Path $control 'audit.stdout.txt'; $stderr = Join-Path $control 'audit.stderr.txt'
New-Item -ItemType Directory -Force -Path $control, $auditRoot | Out-Null
function Save-State([hashtable] $value) { ($value | ConvertTo-Json -Depth 12 -Compress) | Set-Content -LiteralPath $state -Encoding ascii }
try {
  if (Test-Path -LiteralPath $output) { throw "audit output already exists: $output" }
  if (-not (Test-Path -LiteralPath (Join-Path $heldoutRoot 'receipt.json'))) { throw 'heldout receipt missing' }
  $env:PYTHONPATH = $source
  & python -u -m pure_integer_ai.experiments.audit_free_dialogue_qualification --database $database --heldout-root $heldoutRoot --source-root $source --output $output *> $stdout
  $auditExit = $LASTEXITCODE
  if (-not (Test-Path -LiteralPath $output)) { throw 'qualification output missing' }
  $audit = Get-Content -Raw -LiteralPath $output | ConvertFrom-Json
  $status = if ($audit.qualification_status -eq 'PASS_HELDOUT_CONSUMPTION_ONLY') { 'PASS' } else { 'BLOCKED' }
  Save-State @{format='PURE_INTEGER_BROAD_STRUCTURAL_QUALIFICATION_V1';status=$status;completed_utc=[DateTime]::UtcNow.ToString('o');qualification=$output;audit_exit=$auditExit;free_dialogue_complete=0}
  if ($status -ne 'PASS') { exit 2 }
} catch { $_ | Out-String | Set-Content -LiteralPath $stderr -Encoding utf8; Save-State @{format='PURE_INTEGER_BROAD_STRUCTURAL_QUALIFICATION_V1';status='FAIL';completed_utc=[DateTime]::UtcNow.ToString('o');qualification=$output;error=$_.Exception.Message;free_dialogue_complete=0}; exit 1 }
