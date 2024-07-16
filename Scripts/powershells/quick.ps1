# File: setup-dev-env.ps1

# Set the project root directory
$projectRoot = "C:\TheTradingRobotPlug"

# Set the virtual environment name
$venvName = "TheTradingRobotPlug"

# Set the path to the virtual environment
$venvPath = "$projectRoot\$venvName"

# Set the path for the data bank directory
$dataBankDir = "C:\TheTradingRobotPlugDataBank"

# Set the path for the fallback directory
$fallbackDir = "$dataBankDir\no_date"

# Function to parse date from filename (assumes filenames contain dates in the format YYYY-MM-DD)
function Get-DateFromFilename {
    param (
        [string]$filename
    )

    if ($filename -match "\d{4}-\d{2}-\d{2}") {
        return [datetime]::Parse($matches[0])
    }
    return $null
}

# Function to extract the first date from a CSV file
function Get-DateFromCSV {
    param (
        [string]$filePath
    )

    try {
        $content = Get-Content -Path $filePath -First 2
        $headers = $content[0] -split ','
        $data = $content[1] -split ','

        for ($i = 0; $i -lt $headers.Length; $i++) {
            if ($data[$i] -match "\d{4}-\d{2}-\d{2}") {
                return [datetime]::Parse($data[$i])
            }
        }
    } catch {
        Write-Error "Failed to read date from CSV file ${filePath}: $($_.Exception.Message)"
    }
    return $null
}

# Function to move CSV files to the data bank with organization
function Move-DataToDataBank {
    param (
        [string]$sourceDir,
        [string]$targetDir,
        [string]$fallbackDir
    )

    # Ensure the target directory and fallback directory exist
    if (-Not (Test-Path $targetDir)) {
        try {
            New-Item -ItemType Directory -Path $targetDir -Force
            Write-Host "Created data bank directory at $targetDir"
        } catch {
            $errorMessage = "Failed to create data bank directory at ${targetDir}: $($_.Exception.Message)"
            Write-Error $errorMessage
            return
        }
    }

    if (-Not (Test-Path $fallbackDir)) {
        try {
            New-Item -ItemType Directory -Path $fallbackDir -Force
            Write-Host "Created fallback directory at $fallbackDir"
        } catch {
            $errorMessage = "Failed to create fallback directory at ${fallbackDir}: $($_.Exception.Message)"
            Write-Error $errorMessage
            return
        }
    }

    # Get all CSV files in the source directory
    $csvFiles = Get-ChildItem -Path $sourceDir -Recurse -Filter *.csv

    if ($csvFiles.Count -eq 0) {
        Write-Host "No CSV files found in $sourceDir."
        return
    }

    # Move each CSV file to the target directory or fallback directory
    foreach ($file in $csvFiles) {
        $date = Get-DateFromFilename -filename $file.Name
        if ($date -eq $null) {
            $date = Get-DateFromCSV -filePath $file.FullName
        }

        if ($date -ne $null) {
            $year = $date.Year
            $month = $date.ToString("MM")
            $source = $file.Directory.Parent.Name
            $symbol = $file.BaseName.Split("_")[0]  # Assumes symbol is the first part of the filename

            $newFileName = "$symbol_$($date.ToString('yyyy-MM-dd'))$($file.Extension)"
            $destinationDir = Join-Path -Path $targetDir -ChildPath "$source\$symbol\$year\$month"
            if (-Not (Test-Path $destinationDir)) {
                try {
                    New-Item -ItemType Directory -Path $destinationDir -Force
                } catch {
                    $errorMessage = "Failed to create directory ${destinationDir}: $($_.Exception.Message)"
                    Write-Error $errorMessage
                    continue
                }
            }

            $destinationPath = Join-Path -Path $destinationDir -ChildPath $newFileName
            try {
                Move-Item -Path $file.FullName -Destination $destinationPath -Force
                Write-Host "Moved $($file.FullName) to $destinationPath"
            } catch {
                $errorMessage = "Failed to move $($file.FullName) to ${destinationPath}: $($_.Exception.Message)"
                Write-Error $errorMessage
            }
        } else {
            $currentDate = Get-Date
            $newFileName = "$($file.BaseName)_$($currentDate.ToString('yyyy-MM-dd'))$($file.Extension)"
            $fallbackPath = Join-Path -Path $fallbackDir -ChildPath $newFileName
            try {
                Move-Item -Path $file.FullName -Destination $fallbackPath -Force
                Write-Host "Moved $($file.FullName) to fallback directory $fallbackPath"
            } catch {
                $errorMessage = "Failed to move $($file.FullName) to fallback directory ${fallbackPath}: $($_.Exception.Message)"
                Write-Error $errorMessage
            }
        }
    }
}

# Create the virtual environment if it doesn't exist
if (-Not (Test-Path $venvPath)) {
    try {
        python -m venv $venvPath
        Write-Host "Virtual environment created at $venvPath"
    } catch {
        $errorMessage = "Failed to create virtual environment at ${venvPath}: $($_.Exception.Message)"
        Write-Error $errorMessage
        exit 1
    }
} else {
    Write-Host "Virtual environment already exists at $venvPath"
}

# Activate the virtual environment
$activateScript = "$venvPath\Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated."
} else {
    Write-Error "Activation script not found at ${activateScript}"
    exit 1
}

# Install required packages
if (Test-Path "$projectRoot\requirements.txt") {
    try {
        pip install -r "$projectRoot\requirements.txt"
        Write-Host "Installed packages from requirements.txt"
    } catch {
        $errorMessage = "Failed to install packages from requirements.txt: $($_.Exception.Message)"
        Write-Error $errorMessage
        exit 1
    }
} else {
    Write-Host "requirements.txt not found. Please ensure it exists in the project root."
}

# Ask if the user wants to move data to the data bank
$moveData = Read-Host "Would you like to move CSV data to the data bank? (yes/no)"

if ($moveData -eq "yes") {
    # Move CSV files from the data directories to the data bank
    $dataDirs = @(
        "$projectRoot\data\alpha_vantage",
        "$projectRoot\data\csv",
        "$projectRoot\data\polygon",
        "$projectRoot\data\processed",
        "$projectRoot\data\raw"
    )

    foreach ($dir in $dataDirs) {
        if (Test-Path $dir) {
            Move-DataToDataBank -sourceDir $dir -targetDir $dataBankDir -fallbackDir $fallbackDir
        } else {
            Write-Host "Directory $dir does not exist."
        }
    }

    Write-Host "Data move completed."
} else {
    Write-Host "Data move skipped."
}

# Start PowerShell with the virtual environment activated
Write-Host "Starting PowerShell with the virtual environment activated..."
powershell -NoExit -Command "& $activateScript"
