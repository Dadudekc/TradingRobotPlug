# Define the root directory
$rootDir = "C:\TheTradingRobotPlug"

# Function to list files and directories in a tree structure
function Get-DirectoryStructure {
    param (
        [string]$path,
        [string]$indent = ""
    )
    
    # Get all items in the current directory
    $items = Get-ChildItem -Path $path

    foreach ($item in $items) {
        if ($item.PSIsContainer) {
            Write-Output "$indent+-- $($item.Name)"
            Get-DirectoryStructure -path $item.FullName -indent "$indent|   "
        } else {
            Write-Output "$indent+-- $($item.Name)"
        }
    }
}

# Start the tree structure with the root directory
Write-Output $rootDir
Get-DirectoryStructure -path $rootDir -indent ""
