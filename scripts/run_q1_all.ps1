param(
  [string]$DataRoot = "./data",
  [switch]$UseWandb,
  [switch]$DryRun
)

function Invoke-Step {
  param(
    [string]$Exe,
    [string[]]$StepArgs,
    [bool]$SkipExecution
  )

  $display = "$Exe " + ($StepArgs -join " ")
  Write-Host $display
  if (-not $SkipExecution) {
    & $Exe @StepArgs
  }
}

$pythonExe = "python"
if (Test-Path ".\.venv311\Scripts\python.exe") {
  $pythonExe = ".\.venv311\Scripts\python.exe"
} elseif (Test-Path ".\.venv\Scripts\python.exe") {
  $pythonExe = ".\.venv\Scripts\python.exe"
}

$baseArgs = @("--data_root", $DataRoot)

$baselineArgs = @("-m", "src.q1_vit_lora.train_baseline") + $baseArgs + @("--output_dir", "results/q1/baseline")
if ($UseWandb) { $baselineArgs += "--use_wandb" }
Invoke-Step -Exe $pythonExe -StepArgs $baselineArgs -SkipExecution:$DryRun

$gridArgs = @("-m", "src.q1_vit_lora.run_grid") + $baseArgs + @("--output_root", "results/q1/lora_grid", "--weights_root", "weights/q1/grid")
if ($UseWandb) { $gridArgs += "--use_wandb" }
Invoke-Step -Exe $pythonExe -StepArgs $gridArgs -SkipExecution:$DryRun

$optunaArgs = @("-m", "src.q1_vit_lora.optuna_lora_search") + $baseArgs + @("--output_dir", "results/q1/optuna")
Invoke-Step -Exe $pythonExe -StepArgs $optunaArgs -SkipExecution:$DryRun
