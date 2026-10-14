$ErrorActionPreference = "Stop"

$ProjectRoot = "D:\大四上\AIDR\Project"
$TransformerRoot = Join-Path $ProjectRoot "CoT_effectiveness_research\pilot\transformer"
$RunRoot = Join-Path $TransformerRoot "runs"
$Python = "C:\ProgramData\anaconda3\python.exe"
$DatasetDir = Join-Path $ProjectRoot "mathematics_dataset\data\clean-100k"
$TokenizerDir = Join-Path $ProjectRoot "CoT_effectiveness_research\pilot\tokenizer\tokenizer\v1_sp_bpe_8k_seed1234"
$LogRoot = Join-Path $RunRoot ("lr_sweep_logs_" + (Get-Date -Format "yyyyMMdd_HHmmss"))

$Experiments = @(
    [pscustomobject]@{
        Name         = "baseline_eos_lr1e-4_20ep"
        LearningRate = "1e-4"
    },
    [pscustomobject]@{
        Name         = "baseline_eos_lr5e-5_20ep"
        LearningRate = "5e-5"
    }
)

# Check both runs before starting either one, so the first experiment cannot
# finish only for the second output directory to cause an overwrite refusal.
foreach ($Path in @($Python, (Join-Path $DatasetDir "train.jsonl"), (Join-Path $DatasetDir "valid.jsonl"), (Join-Path $TokenizerDir "tokenizer.model"))) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required file not found: $Path"
    }
}

foreach ($Experiment in $Experiments) {
    $OutputDir = Join-Path $RunRoot $Experiment.Name
    if ((Test-Path -LiteralPath $OutputDir) -and (Get-ChildItem -Force -LiteralPath $OutputDir | Select-Object -First 1)) {
        throw "Refusing to overwrite non-empty run directory: $OutputDir"
    }
}

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null
$env:PATH = @(
    "C:\ProgramData\anaconda3\Library\bin"
    "C:\ProgramData\anaconda3\Library\usr\bin"
    "C:\ProgramData\anaconda3\Library\mingw-w64\bin"
    $env:PATH
) -join ";"
Set-Location -LiteralPath $ProjectRoot

$Results = [System.Collections.Generic.List[object]]::new()

foreach ($Experiment in $Experiments) {
    $OutputDir = Join-Path $RunRoot $Experiment.Name
    $LogPath = Join-Path $LogRoot ($Experiment.Name + ".log")

    Write-Host "Starting $($Experiment.Name) (learning rate $($Experiment.LearningRate))"
    Write-Host "Run output: $OutputDir"
    Write-Host "Console log: $LogPath"

    $TrainArgs = @(
        "-m", "CoT_effectiveness_research.pilot.transformer.train",
        "--dataset-dir", $DatasetDir,
        "--tokenizer-dir", $TokenizerDir,
        "--output-dir", $OutputDir,
        "--epochs", "20",
        "--learning-rate", $Experiment.LearningRate,
        "--batch-size", "32",
        "--eos-loss-weight", "1.0",
        "--seed", "1234",
        "--epoch-valid-per-type", "100",
        "--skip-eval"
    )

    # Do not merge native stderr into the PowerShell error stream.  PyTorch
    # emits a harmless UserWarning during Transformer construction; Windows
    # PowerShell otherwise turns that stderr line into a terminating
    # NativeCommandError when ErrorActionPreference=Stop is enabled.
    $StdoutPath = Join-Path $LogRoot ($Experiment.Name + ".stdout.tmp")
    $StderrPath = Join-Path $LogRoot ($Experiment.Name + ".stderr.tmp")
    $Process = Start-Process -FilePath $Python -ArgumentList $TrainArgs `
        -WorkingDirectory $ProjectRoot -RedirectStandardOutput $StdoutPath `
        -RedirectStandardError $StderrPath -Wait -PassThru
    if (Test-Path -LiteralPath $StdoutPath) {
        Copy-Item -LiteralPath $StdoutPath -Destination $LogPath -Force
    } else {
        New-Item -ItemType File -Path $LogPath -Force | Out-Null
    }
    if (Test-Path -LiteralPath $StderrPath) {
        Add-Content -LiteralPath $LogPath -Value "`n===== native stderr ====="
        Get-Content -LiteralPath $StderrPath | Add-Content -LiteralPath $LogPath
    }
    Remove-Item -LiteralPath $StdoutPath, $StderrPath -Force -ErrorAction SilentlyContinue
    $ExitCode = $Process.ExitCode
    if ($ExitCode -ne 0) {
        throw "$($Experiment.Name) failed with exit code $ExitCode. See log: $LogPath"
    }

    $ManifestPath = Join-Path $OutputDir "run_manifest.json"
    if (-not (Test-Path -LiteralPath $ManifestPath)) {
        throw "Training returned success but the run manifest is missing: $ManifestPath"
    }
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw | ConvertFrom-Json
    if (-not $Manifest.training_complete) {
        throw "$($Experiment.Name) ended without training_complete=true. See manifest: $ManifestPath"
    }

    $Results.Add([pscustomobject]@{
            run                   = $Experiment.Name
            learning_rate         = $Experiment.LearningRate
            completed_epochs      = $Manifest.completed_epochs
            best_epoch            = $Manifest.best_epoch
            best_valid_answer_nll = $Manifest.best_valid_answer_nll
            best_checkpoint       = $Manifest.best_checkpoint_path
            final_checkpoint      = $Manifest.checkpoint_path
            log                   = $LogPath
        })
}

$SummaryPath = Join-Path $LogRoot "summary.json"
$Results | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $SummaryPath -Encoding UTF8

Write-Host "Both runs completed. Summary: $SummaryPath"
$Results | Format-Table -AutoSize
