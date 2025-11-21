Param(
    [string]$Root = $PWD.Path
)

# Create folders and set user-level environment variable (persists across sessions)
$folders = @(
    "configs",
    "data\raw",
    "data\interim",
    "data\processed",
    "artifacts\figures",
    "artifacts\models",
    "artifacts\metrics",
    "artifacts\tables",
    "artifacts\manifests",
    "logs",
    "notebooks",
    "reports",
    "scripts",
    "src\cntlab"
)

foreach ($f in $folders) {
    $p = Join-Path $Root $f
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p | Out-Null
    }
}

# Set user environment var
[Environment]::SetEnvironmentVariable("CNT_LAB_DIR", $Root, "User")
Write-Host "CNT_LAB_DIR set to $Root (User scope)"
Write-Host "Folders ensured."
