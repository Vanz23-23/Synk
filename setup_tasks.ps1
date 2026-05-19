# Run this once as Administrator (right-click PowerShell -> Run as administrator)
# Sets up Task Scheduler for Synk bot + watchdog + weekly review.
# Paths are resolved relative to this script's location, so the repo can live anywhere.

$synkDir  = $PSScriptRoot
$pythonW  = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $pythonW) {
    Write-Error "pythonw not found on PATH. Install Python and ensure it is on PATH, or edit this script to set an absolute path."
    exit 1
}

# --- SynkBot: runs at logon, loops forever (restart loop is inside the .bat) ---
$botAction   = New-ScheduledTaskAction -Execute "$synkDir\run_bot.bat" -WorkingDirectory $synkDir
$botTrigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$botSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "SynkBot" -Action $botAction -Trigger $botTrigger `
    -Settings $botSettings -RunLevel Highest -Force
Write-Host "SynkBot task registered."

# --- SynkWatchdog: runs every 15 minutes (pythonw = no console window) ---
$wdAction   = New-ScheduledTaskAction -Execute $pythonW `
    -Argument "alerts\watchdog.py" -WorkingDirectory $synkDir
$wdTrigger  = New-ScheduledTaskTrigger -RepetitionInterval (New-TimeSpan -Minutes 15) -Once -At (Get-Date)
$wdSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "SynkWatchdog" -Action $wdAction -Trigger $wdTrigger `
    -Settings $wdSettings -RunLevel Highest -Force
Write-Host "SynkWatchdog task registered."

# --- SynkWeeklyReview: runs every Sunday at 19:00 (pythonw = no console window) ---
$reviewAction   = New-ScheduledTaskAction -Execute $pythonW `
    -Argument "weekly_review.py" -WorkingDirectory $synkDir
$reviewTrigger  = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "19:00"
$reviewSettings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 5)
Register-ScheduledTask -TaskName "SynkWeeklyReview" -Action $reviewAction `
    -Trigger $reviewTrigger -Settings $reviewSettings -RunLevel Highest -Force
Write-Host "SynkWeeklyReview task registered."

Write-Host ""
Write-Host "Done. Bot will start automatically on next logon."
Write-Host "To start now without logging off: Start-ScheduledTask -TaskName SynkBot"
