#requires -Version 5.1
<#
.SYNOPSIS
    Merge webcam samples with imported Kaggle landmarks, retrain, select, and evaluate.

.DESCRIPTION
    One-command real-data pipeline:

        1. (optional) import Kaggle images -> data/imported/landmarks.csv
        2. merge data/raw/landmarks.csv with the imported landmarks
        3. create leakage-aware splits in data/processed_kaggle/
        4. train SVM and Random Forest candidates
        5. copy the better model (validation macro-F1) to models/classifier.joblib
        6. evaluate once on the held-out test split

    The synthetic demo model (models/demo_classifier.joblib) is never touched.

.PARAMETER MaxPerClass
    Maximum Kaggle images sampled per class when the import is first created.

.PARAMETER RefreshImport
    Re-run the importer even if data/imported/landmarks.csv already exists.

.PARAMETER SkipImport
    Never run the importer. The existing imported CSV must be present.

.PARAMETER Tune
    Enable the slower grouped hyperparameter search for both candidates.

.PARAMETER TestSubjects
    Subject ids to hold out for test, for example -TestSubjects me. If omitted,
    the test split is grouped by pseudo-session but is NOT subject-disjoint.

.PARAMETER SkipEvaluate
    Train and select a model without running evaluate_model.py.

.PARAMETER StartApp
    Start the FastAPI app with the selected model after evaluation.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_real.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\run_real.ps1 -TestSubjects me -StartApp
#>
[CmdletBinding()]
param(
    [int]$MaxPerClass = 1500,
    [switch]$RefreshImport,
    [switch]$SkipImport,
    [switch]$Tune,
    [string[]]$TestSubjects = @(),
    [switch]$SkipEvaluate,
    [switch]$StartApp,
    [int]$Port = 5055
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ImportedCsv = Join-Path $ProjectRoot "data\imported\landmarks.csv"
$RawCsv = Join-Path $ProjectRoot "data\raw\landmarks.csv"
$MergedCsv = Join-Path $ProjectRoot "data\merged\landmarks.csv"
$ProcessedDir = Join-Path $ProjectRoot "data\processed_kaggle"
$DatasetRoot = Join-Path $ProjectRoot "Dataset_kaggle\asl_alphabet_train"

function Invoke-PipelineStep {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Host ""
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    Write-Host $Label -ForegroundColor Cyan
    Write-Host ("=" * 78) -ForegroundColor DarkGray
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

Write-Host "Sign Language Recognition - real-data pipeline" -ForegroundColor Green
Write-Host "Project: $ProjectRoot"
Write-Host "Python:  $Python"

# ---------------------------------------------------------------- import
if (-not $SkipImport) {
    if ((Test-Path $ImportedCsv) -and -not $RefreshImport) {
        Write-Host "Using existing imported landmarks: $ImportedCsv" -ForegroundColor Yellow
        Write-Host "Pass -RefreshImport to rebuild it from the Kaggle images."
    }
    else {
        if (-not (Test-Path $DatasetRoot)) {
            throw "Kaggle dataset not found at $DatasetRoot. Pass -SkipImport to use an existing import."
        }
        Invoke-PipelineStep "Import Kaggle images -> landmarks" @(
            "scripts\import_image_dataset.py",
            "--root", $DatasetRoot,
            "--max-per-class", "$MaxPerClass",
            "--output", $ImportedCsv,
            "--overwrite"
        )
    }
}
else {
    Write-Host "Import skipped (-SkipImport)." -ForegroundColor Yellow
}

if (-not (Test-Path $ImportedCsv)) {
    throw "Missing imported landmarks: $ImportedCsv"
}

# ---------------------------------------------------------------- merge
if (Test-Path $RawCsv) {
    Write-Host "Webcam samples: $RawCsv" -ForegroundColor Green
}
else {
    Write-Host "No webcam samples yet: $RawCsv" -ForegroundColor Yellow
    Write-Host "Collect some with: python collect_data.py --label A --subject me --session s01"
}
Invoke-PipelineStep "Merge webcam + imported landmarks" @(
    "scripts\merge_datasets.py",
    "--imported", $ImportedCsv,
    "--raw", $RawCsv,
    "--out", $MergedCsv
)

# ---------------------------------------------------------------- preprocess
$PreprocessArgs = @(
    "preprocess_data.py",
    "--input", $MergedCsv,
    "--out", $ProcessedDir
)
if ($TestSubjects.Count -gt 0) {
    $PreprocessArgs += "--test-subjects"
    $PreprocessArgs += $TestSubjects
}
Invoke-PipelineStep "Validate and split dataset" $PreprocessArgs

# ---------------------------------------------------------------- train candidates
$TuneArgs = @()
if (-not $Tune) {
    $TuneArgs += "--no-tune"
}

$Candidates = @(
    @{ Name = "svm"; Model = "models\candidate_svm.joblib"; Card = "models\candidate_svm_card.json" },
    @{ Name = "rf";  Model = "models\candidate_rf.joblib";  Card = "models\candidate_rf_card.json" }
)

foreach ($Candidate in $Candidates) {
    $TrainArgs = @(
        "train_model.py",
        "--processed", $ProcessedDir,
        "--model", $Candidate.Name,
        "--out", $Candidate.Model,
        "--augment-copies", "1",
        "--experiment-log", "artifacts\experiments.jsonl"
    ) + $TuneArgs
    Invoke-PipelineStep "Train $($Candidate.Name.ToUpper()) candidate" $TrainArgs
}

# ---------------------------------------------------------------- select
$Results = @()
foreach ($Candidate in $Candidates) {
    if (-not (Test-Path $Candidate.Card)) {
        throw "Missing model card: $($Candidate.Card)"
    }
    $Card = Get-Content -Raw -Path $Candidate.Card | ConvertFrom-Json
    $F1 = [double]$Card.validation_metrics.macro_f1
    $Accuracy = [double]$Card.validation_metrics.accuracy
    $Results += [pscustomobject]@{
        Name = $Candidate.Name.ToUpper()
        MacroF1 = $F1
        Accuracy = $Accuracy
        Model = $Candidate.Model
        Card = $Candidate.Card
    }
}

$Best = $Results | Sort-Object -Property MacroF1 -Descending | Select-Object -First 1
Write-Host ""
Write-Host "Candidate comparison (validation only)" -ForegroundColor Cyan
$Results | Format-Table Name, MacroF1, Accuracy -AutoSize

Copy-Item -Path $Best.Model -Destination "models\classifier.joblib" -Force
Copy-Item -Path $Best.Card -Destination "models\classifier_card.json" -Force

# Keep the copied card's filename honest about the selected bundle.
$FinalCard = Get-Content -Raw -Path "models\classifier_card.json" | ConvertFrom-Json
$FinalCard.model_file = "classifier.joblib"
$FinalCard | ConvertTo-Json -Depth 30 | Set-Content -Path "models\classifier_card.json" -Encoding UTF8

$Selection = [pscustomobject]@{
    selected_utc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
    selected_model = $Best.Name
    validation_macro_f1 = $Best.MacroF1
    validation_accuracy = $Best.Accuracy
    candidates = $Results
    warning = "Test split was not used for selection. Imported pseudo-sessions are not subject-disjoint unless -TestSubjects was supplied."
}
$Selection | ConvertTo-Json -Depth 5 | Set-Content -Path "models\classifier_selection.json" -Encoding UTF8

Write-Host ""
Write-Host "Selected: $($Best.Name) (validation macro-F1 $($Best.MacroF1.ToString('0.0000')))" -ForegroundColor Green
Write-Host "Model:    models\classifier.joblib"
Write-Host "Card:     models\classifier_card.json"

# ---------------------------------------------------------------- evaluate
if (-not $SkipEvaluate) {
    Invoke-PipelineStep "Evaluate selected model on the test split" @(
        "evaluate_model.py",
        "--processed", $ProcessedDir,
        "--model", "models\classifier.joblib",
        "--split", "test",
        "--output", "artifacts\eval_test_real"
    )
}
else {
    Write-Host "Evaluation skipped (-SkipEvaluate)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "Report: artifacts\eval_test_real\report.md"
Write-Host "Live app: python app.py --model models/classifier.joblib --port $Port --autostart"

if ($StartApp) {
    Start-Process -FilePath $Python -WorkingDirectory $ProjectRoot -ArgumentList @(
        "app.py", "--model", "models\classifier.joblib", "--port", "$Port", "--autostart"
    )
    Write-Host "Started app at http://127.0.0.1:$Port" -ForegroundColor Green
}
