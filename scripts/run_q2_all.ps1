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

$trainArgs = @("-m", "src.q2_attacks.train_clean_resnet18") + $baseArgs + @("--output_dir", "results/q2/clean_resnet18")
if ($UseWandb) { $trainArgs += "--use_wandb" }
Invoke-Step -Exe $pythonExe -StepArgs $trainArgs -SkipExecution:$DryRun

$fgsmArgs = @("-m", "src.q2_attacks.fgsm_compare") + $baseArgs + @("--output_dir", "results/q2/fgsm_compare")
if ($UseWandb) { $fgsmArgs += "--use_wandb" }
Invoke-Step -Exe $pythonExe -StepArgs $fgsmArgs -SkipExecution:$DryRun

$pgdArgs = @("-m", "src.q2_detection.train_detector") + $baseArgs + @("--attack", "pgd", "--output_dir", "results/q2/detectors", "--weights_dir", "weights/q2/detectors")
if ($UseWandb) { $pgdArgs += "--use_wandb" }
Invoke-Step -Exe $pythonExe -StepArgs $pgdArgs -SkipExecution:$DryRun

$bimArgs = @("-m", "src.q2_detection.train_detector") + $baseArgs + @("--attack", "bim", "--output_dir", "results/q2/detectors", "--weights_dir", "weights/q2/detectors")
if ($UseWandb) { $bimArgs += "--use_wandb" }
Invoke-Step -Exe $pythonExe -StepArgs $bimArgs -SkipExecution:$DryRun

if ($UseWandb) {
  $sampleArgs = @("-m", "src.q2_attacks.log_all_attack_samples_wandb") + $baseArgs
  Invoke-Step -Exe $pythonExe -StepArgs $sampleArgs -SkipExecution:$DryRun
}
