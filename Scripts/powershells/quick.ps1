# Define the root path of your project
$projectRoot = "C:\TheTradingRobotPlug"

# Get all directories recursively, excluding the Documents directory and its subdirectories
$directories = Get-ChildItem -Path $projectRoot -Directory -Recurse | Where-Object {
    $_.FullName -notmatch "\\Documents($|\\)"
}

# Function to add __init__.py file if it doesn't exist
function Add-Init-File {
    param (
        [string]$directory
    )

    $initFilePath = Join-Path -Path $directory -ChildPath "__init__.py"
    if (-Not (Test-Path -Path $initFilePath)) {
        New-Item -ItemType File -Path $initFilePath -Force
        Write-Output "Created: $initFilePath"
    } else {
        Write-Output "__init__.py already exists in: $directory"
    }
}

# Add __init__.py to each directory
foreach ($dir in $directories) {
    Add-Init-File -directory $dir.FullName
}

Write-Output "Completed adding __init__.py files."
