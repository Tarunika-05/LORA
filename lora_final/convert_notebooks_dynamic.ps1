param(
    [Parameter(Mandatory=$true)]
    [string]$SourceFile,
    
    [Parameter(Mandatory=$true)]
    [string]$OutputFile
)

$fullContent = Get-Content -LiteralPath $SourceFile -Raw

# Split on # %% markers
$sections = $fullContent -split '(?m)^# %%'
$sections = $sections | Where-Object { $_.Trim() -ne '' }

$cells = @()

foreach ($section in $sections) {
    if ([string]::IsNullOrWhiteSpace($section)) { continue }
    $lines = $section -split "`n"
    
    # Check if it's a markdown cell
    if ($lines[0].Trim() -eq '[markdown]') {
        # Extract markdown content (remove leading # from each line)
        $mdLines = @()
        for ($i = 1; $i -lt $lines.Count; $i++) {
            $line = $lines[$i]
            if ($line -match '^# (.*)') {
                $mdLines += $Matches[1] + "`n"
            } elseif ($line.Trim() -eq '#') {
                $mdLines += "`n"
            } elseif ($line.Trim() -eq '') {
                # skip empty lines between sections
            }
        }
        if ($mdLines.Count -gt 0) {
            $cell = @{
                cell_type = "markdown"
                metadata = @{ id = [guid]::NewGuid().ToString().Substring(0,12) }
                source = $mdLines
            }
            $cells += $cell
        }
    } else {
        # Code cell
        $codeLines = @()
        foreach ($line in $lines) {
            $codeLines += $line + "`n"
        }
        # Remove leading empty line if present
        if ($codeLines.Count -gt 0 -and $codeLines[0].Trim() -eq '') {
            $codeLines = $codeLines[1..($codeLines.Count-1)]
        }
        if ($codeLines.Count -gt 0) {
            $cell = @{
                cell_type = "code"
                execution_count = $null
                metadata = @{ id = [guid]::NewGuid().ToString().Substring(0,12) }
                outputs = @()
                source = $codeLines
            }
            $cells += $cell
        }
    }
}

$notebook = @{
    nbformat = 4
    nbformat_minor = 0
    metadata = @{
        colab = @{
            provenance = @()
            gpuType = "T4"
        }
        kernelspec = @{
            name = "python3"
            display_name = "Python 3"
        }
        language_info = @{
            name = "python"
        }
        accelerator = "GPU"
    }
    cells = $cells
}

$json = $notebook | ConvertTo-Json -Depth 10
$json | Out-File -LiteralPath $OutputFile -Encoding utf8

Write-Host "Notebook $OutputFile created successfully with $($cells.Count) cells"
