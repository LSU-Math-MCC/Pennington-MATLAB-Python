Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-CleanDestination {
    param([string] $Path)
    if (Test-Path -LiteralPath $Path) {
        $item = Get-Item -LiteralPath $Path -Force
        if (-not $item.PSIsContainer) {
            throw "Destination exists as a file: $Path"
        }
        $children = @(Get-ChildItem -LiteralPath $Path -Force)
        if ($children.Count -gt 0) {
            throw "Destination directory is not empty: $Path"
        }
    }
}

function Invoke-GitMv {
    param(
        [string] $Source,
        [string] $Destination
    )
    if (-not (Test-Path -LiteralPath $Source)) {
        throw "Source path is missing: $Source"
    }
    Assert-CleanDestination $Destination
    $parent = Split-Path -Parent $Destination
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    & git mv $Source $Destination
    if ($LASTEXITCODE -ne 0) {
        throw "git mv failed: $Source -> $Destination"
    }
}

function Write-Breadcrumb {
    param(
        [string] $Directory,
        [string] $OldPath,
        [string] $NewPath,
        [string] $OldNative,
        [string] $NewNative,
        [string] $UnifiedUsage
    )
    if (-not (Test-Path -LiteralPath $Directory)) {
        New-Item -ItemType Directory -Force -Path $Directory | Out-Null
    }
    $readme = @"
# Relocated

The project formerly stored at $OldPath was relocated verbatim to $NewPath.

Its original README is now:

$NewPath/README.md

Old native usage:

$OldNative

New native usage:

$NewNative

New unified usage:

$UnifiedUsage

Do not add new implementation files here. Update the relocated project instead.
See unified/RELOCATION_MAP.md for the complete mapping.
"@
    Set-Content -LiteralPath (Join-Path $Directory "README.md") -Value $readme -Encoding UTF8
    & git add (Join-Path $Directory "README.md")
    if ($LASTEXITCODE -ne 0) {
        throw "git add failed for breadcrumb: $Directory"
    }
}

$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
$head = (& git rev-parse HEAD).Trim()
$status = & git status --short
$trackedBefore = @(& git ls-files -- Python_img_to_obj Python_Fall2025 Python_ML_2021 unified)

Assert-CleanDestination "unified/img2obj"
Assert-CleanDestination "unified/obj2anthro"
Assert-CleanDestination "unified/ml"

Invoke-GitMv "unified/pipeline.py" "unified/obj2anthro/pipeline.py"
Invoke-GitMv "unified/schema.py" "unified/obj2anthro/schema.py"
Invoke-GitMv "unified/backends.py" "unified/obj2anthro/backend_registry.py"
Invoke-GitMv "unified/__main__.py" "unified/obj2anthro/cli.py"
Invoke-GitMv "unified/tests" "unified/obj2anthro/tests"
Invoke-GitMv "unified/results" "unified/obj2anthro/results"
Invoke-GitMv "unified/README.md" "unified/obj2anthro/README.md"

Invoke-GitMv "Python_Fall2025" "unified/obj2anthro/backends/segmentation"
Invoke-GitMv "Python_img_to_obj" "unified/img2obj"
Invoke-GitMv "Python_ML_2021" "unified/ml/experiment"

Write-Breadcrumb `
    -Directory "Python_img_to_obj" `
    -OldPath "Python_img_to_obj/" `
    -NewPath "unified/img2obj" `
    -OldNative "cd Python_img_to_obj && python -m pipeline.run single --image IMG --out OUT" `
    -NewNative "cd unified/img2obj && python -m pipeline.run single --image IMG --out OUT" `
    -UnifiedUsage "python -m unified img2obj --input IMG --out OUT"

Write-Breadcrumb `
    -Directory "Python_Fall2025" `
    -OldPath "Python_Fall2025/" `
    -NewPath "unified/obj2anthro/backends/segmentation" `
    -OldNative "cd Python_Fall2025 && python -m src.main ..." `
    -NewNative "cd unified/obj2anthro/backends/segmentation && python -m src.main ..." `
    -UnifiedUsage "python -m unified obj2anthro --input OBJ --method segmentation"

Write-Breadcrumb `
    -Directory "Python_ML_2021" `
    -OldPath "Python_ML_2021/" `
    -NewPath "unified/ml/experiment" `
    -OldNative "Use historical scripts under Python_ML_2021/..." `
    -NewNative "Use the same relative scripts under unified/ml/experiment/..." `
    -UnifiedUsage "python -m unified ml"

$trackedAfter = @(& git ls-files -- Python_img_to_obj Python_Fall2025 Python_ML_2021 unified)
$report = @(
    "# Pennington relocation audit",
    "",
    "Branch: $branch",
    "HEAD: $head",
    "",
    "Pre-move status:",
    '```',
    ($status -join [Environment]::NewLine),
    '```',
    "",
    "Tracked files before: $($trackedBefore.Count)",
    "Tracked files after: $($trackedAfter.Count)",
    "",
    "Directory moves:",
    "- Python_img_to_obj/ -> unified/img2obj/",
    "- Python_Fall2025/ -> unified/obj2anthro/backends/segmentation/",
    "- Python_ML_2021/ -> unified/ml/experiment/",
    "- unified anthropometry files -> unified/obj2anthro/",
    "",
    "Old-path breadcrumbs:",
    "- Python_img_to_obj/README.md",
    "- Python_Fall2025/README.md",
    "- Python_ML_2021/README.md"
)
Set-Content -LiteralPath ".git/pennington-relocation-report.md" -Value $report -Encoding UTF8
Write-Host "Relocation script complete. Audit: .git/pennington-relocation-report.md"
