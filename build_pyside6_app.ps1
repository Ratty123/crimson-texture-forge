param(
    [ValidateSet("onedir", "onefile")]
    [string]$Mode = "onefile"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$appName = "CrimsonTextureForge"
$legacyAppName = "DDSRebuildApp"
$iconPath = Join-Path $PSScriptRoot "assets\crimson_texture_forge.ico"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$stableDistDir = Join-Path $scriptDir "dist"
$stableBuildDir = Join-Path $scriptDir "build"
$pyInstallerDistDir = Join-Path $stableBuildDir "pyinstaller-dist"
$pyInstallerWorkDir = Join-Path $stableBuildDir "pyinstaller-work"

function Remove-PathWithRetries {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath,
        [switch]$Recurse,
        [int]$RetryCount = 8,
        [int]$DelayMilliseconds = 400
    )

    if (-not (Test-Path -LiteralPath $LiteralPath)) {
        return
    }

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
        try {
            if ($Recurse) {
                Remove-Item -LiteralPath $LiteralPath -Recurse -Force -ErrorAction Stop
            } else {
                Remove-Item -LiteralPath $LiteralPath -Force -ErrorAction Stop
            }
            return
        } catch {
            if ($attempt -ge $RetryCount) {
                throw "Failed to remove '$LiteralPath' after $RetryCount attempt(s): $($_.Exception.Message)"
            }
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
}

function Move-PathWithRetries {
    param(
        [Parameter(Mandatory = $true)]
        [string]$SourcePath,
        [Parameter(Mandatory = $true)]
        [string]$DestinationPath,
        [int]$RetryCount = 8,
        [int]$DelayMilliseconds = 400
    )

    if (-not (Test-Path -LiteralPath $SourcePath)) {
        throw "Source path does not exist: $SourcePath"
    }

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
        try {
            Move-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force -ErrorAction Stop
            return
        } catch {
            if ($attempt -ge $RetryCount) {
                throw "Failed to move '$SourcePath' to '$DestinationPath' after $RetryCount attempt(s): $($_.Exception.Message)"
            }
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
}

function Stop-AppProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$NamePrefixes
    )

    $targets = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $processName = $_.ProcessName
        foreach ($prefix in $NamePrefixes) {
            if ($processName -like "$prefix*") {
                return $true
            }
        }
        return $false
    } | Sort-Object Id -Unique)

    if (-not $targets) {
        return
    }

    Write-Host "Stopping running build targets..."
    foreach ($proc in $targets) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction Stop
        } catch {
            Write-Warning "Could not stop process $($proc.ProcessName) [$($proc.Id)]: $($_.Exception.Message)"
        }
    }

    foreach ($proc in $targets) {
        try {
            Wait-Process -Id $proc.Id -Timeout 10 -ErrorAction Stop
        } catch {
            if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
                throw "Process '$($proc.ProcessName)' [$($proc.Id)] is still running after stop was requested."
            }
        }
    }
}

$pythonExe = Join-Path $scriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

$appVersion = (& $pythonExe -c "from crimson_texture_forge.constants import APP_VERSION; print(APP_VERSION)").Trim()
if (-not $appVersion) {
    throw "Could not determine app version from crimson_texture_forge.constants.APP_VERSION"
}
$oneFileOutputName = "$appName-$appVersion-windows-portable.exe"
$oneDirOutputName = "$appName-$appVersion-windows"

Stop-AppProcesses -NamePrefixes @($appName, $legacyAppName)

$pyInstallerArgs = @(
    "-m",
    "PyInstaller",
    "--noconfirm",
    "--clean",
    "--noupx",
    "--windowed",
    "--distpath",
    $pyInstallerDistDir,
    "--workpath",
    $pyInstallerWorkDir,
    "--name",
    $appName
)

if (Test-Path $iconPath) {
    $pyInstallerArgs += @("--icon", $iconPath)
    $pyInstallerArgs += @("--add-data", "$iconPath;assets")
    $pngIconPath = Join-Path $PSScriptRoot "assets\crimson_texture_forge.png"
    if (Test-Path $pngIconPath) {
        $pyInstallerArgs += @("--add-data", "$pngIconPath;assets")
    }
}

if ($Mode -eq "onefile") {
    $pyInstallerArgs += "--onefile"
} else {
    $pyInstallerArgs += "--onedir"
}

New-Item -ItemType Directory -Path $stableDistDir -Force | Out-Null
New-Item -ItemType Directory -Path $stableBuildDir -Force | Out-Null
Remove-PathWithRetries -LiteralPath (Join-Path $stableBuildDir $appName) -Recurse
Remove-PathWithRetries -LiteralPath (Join-Path $stableBuildDir $legacyAppName) -Recurse
Remove-PathWithRetries -LiteralPath $pyInstallerDistDir -Recurse
Remove-PathWithRetries -LiteralPath $pyInstallerWorkDir -Recurse

$pyInstallerArgs += "crimson_texture_forge_app.py"

Write-Host "Building $appName in $Mode mode..."
& $pythonExe @pyInstallerArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($Mode -eq "onefile") {
    $builtExe = Join-Path $pyInstallerDistDir "$appName.exe"
    $versionedExe = Join-Path $stableDistDir $oneFileOutputName
    if (-not (Test-Path $builtExe)) {
        throw "Expected build output not found: $builtExe"
    }
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir "$appName") -Recurse
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir "$appName.exe")
    Remove-PathWithRetries -LiteralPath $versionedExe
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir "$legacyAppName") -Recurse
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir "$legacyAppName.exe")
    Move-PathWithRetries -SourcePath $builtExe -DestinationPath $versionedExe
} else {
    $builtDir = Join-Path $pyInstallerDistDir $appName
    $versionedDir = Join-Path $stableDistDir $oneDirOutputName
    if (-not (Test-Path $builtDir)) {
        throw "Expected build output not found: $builtDir"
    }
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir "$appName.exe")
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir $oneFileOutputName)
    Remove-PathWithRetries -LiteralPath (Join-Path $stableDistDir "$legacyAppName.exe")
    Remove-PathWithRetries -LiteralPath $versionedDir -Recurse
    Move-PathWithRetries -SourcePath $builtDir -DestinationPath $versionedDir
}

Write-Host "Build complete."
if ($Mode -eq "onefile") {
    Write-Host "Output file: $scriptDir\\dist\\$oneFileOutputName"
} else {
    Write-Host "Output folder: $scriptDir\\dist\\$oneDirOutputName"
}
