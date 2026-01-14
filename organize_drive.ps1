<#
.SYNOPSIS
Organizes files in a drive into categorized folders and removes empty directories.
#>

param(
    [string]$Path = 'F:\',
    [string]$Log = 'organize_drive.log',
    [string]$Output = '',  # Optional custom output directory
    [int]$MaxJobs = [Environment]::ProcessorCount
)

$ErrorActionPreference = 'Stop'

Start-Transcript -Path $Log -Append

$Categories = @{
    'Documents' = '.pdf','.doc','.docx','.txt','.xls','.xlsx','.ppt','.pptx','.csv','.odt','.ods','.odp'
    'Images'    = '.jpg','.jpeg','.png','.gif','.bmp','.tiff','.svg'
    'Music'     = '.mp3','.wav','.flac','.aac','.ogg','.wma'
    'Videos'    = '.mp4','.avi','.mkv','.mov','.wmv','.flv'
    'Archives'  = '.zip','.rar','.7z','.tar','.gz','.bz2'
    'Code'      = '.py','.js','.html','.css','.java','.c','.cpp','.cs','.rb','.php'
}

$DefaultCategory = 'Other'
$OrganizedFolder = 'Organized'

function Get-Category {
    param([string]$Extension)
    foreach ($cat in $Categories.Keys) {
        if ($Categories[$cat] -contains $Extension.ToLower()) {
            return $cat
        }
    }
    return $DefaultCategory
}

function Safe-Move {
    param([string]$Source, [string]$Destination)
    $destDir = Split-Path $Destination -Parent
    if (!(Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
    $base = [System.IO.Path]::GetFileNameWithoutExtension($Destination)
    $ext  = [System.IO.Path]::GetExtension($Destination)
    $dest = $Destination
    $counter = 1
    while (Test-Path $dest) {
        $dest = Join-Path $destDir "${base}_${counter}${ext}"
        $counter++
    }
    Move-Item -LiteralPath $Source -Destination $dest
}

function Remove-EmptyDirs {
    param([string]$BasePath)
    Get-ChildItem -Path $BasePath -Directory -Recurse |
        Sort-Object -Property FullName -Descending |
        ForEach-Object {
            if (-not (Get-ChildItem -LiteralPath $_.FullName -Recurse -Force)) {
                try {
                    Remove-Item -LiteralPath $_.FullName -Force
                    Write-Host "Removed empty directory $($_.FullName)"
                } catch {
                    Write-Warning "Failed to remove $($_.FullName): $_"
                }
            }
        }
}

function Move-FileJob {
    param($File, $BaseOutput)
    try {
        $category = Get-Category $File.Extension
        $destDir = Join-Path $BaseOutput $category
        $destination = Join-Path $destDir $File.Name
        Safe-Move -Source $File.FullName -Destination $destination
        Write-Host "Moved $($File.FullName) -> $destination"
    } catch {
        Write-Warning "Failed to move $($File.FullName): $_"
    }
}

$target = Resolve-Path $Path
if ($Output) {
    $baseOutput = Resolve-Path $Output
} else {
    $baseOutput = Join-Path $target $OrganizedFolder
}
if (!(Test-Path $baseOutput)) { New-Item -ItemType Directory -Path $baseOutput | Out-Null }

$files = Get-ChildItem -Path $target -File -Recurse | Where-Object { $_.FullName -notmatch "\\$OrganizedFolder(\\|$)" }
$total = $files.Count
$index = 0

Import-Module ThreadJob -ErrorAction SilentlyContinue
$jobs = @()
foreach ($f in $files) {
    $index++
    $percent = [int](($index / $total) * 100)
    Write-Progress -Activity "Organizing files" -Status "$index of $total" -PercentComplete $percent
    $jobs += Start-ThreadJob -ScriptBlock ${function:Move-FileJob} -ArgumentList $f, $baseOutput
    if ($jobs.Count -ge $MaxJobs) {
        Wait-Job -Job $jobs -Any | Receive-Job
        $jobs = $jobs | Where-Object { $_.State -eq 'Running' }
    }
}

if ($jobs) { Wait-Job $jobs | Receive-Job }
Write-Progress -Activity "Organizing files" -Completed

Remove-EmptyDirs -BasePath $target

Stop-Transcript
Write-Host 'Organization complete.'
