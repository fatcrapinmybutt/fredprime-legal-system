& {
    Set-StrictMode -Version Latest
    $ErrorActionPreference = 'Stop'

    # =========================
    # CONFIG
    # =========================
    $SourceRoots = @(
        'C:\Users\andrew'
    )

    # Change if you want a different destination root
    $DestinationRoot = 'E:\FileHistory\andre\THEMANBEARPIG\_ORGANIZED_BY_EXT_FROM_C_USERS_ANDREW'

    $DoMove = $false                      # $false = COPY, $true = MOVE (COPY recommended initially)
    $Dedupe = $true                       # exact duplicates by SHA-256
    $DedupeAction = 'Quarantine'          # 'Quarantine' | 'Skip' | 'KeepAll'
    $RemoveEmptySourceFolders = $false    # Only consider with MOVE, and only after you verify results
    $SkipReparsePoints = $true            # skips junctions/symlinks to avoid loops
    $ComputeSHA256 = $true                # required for dedupe
    $RunSelfTest = $true                  # validates logic in a sandbox before touching your data
    $SelfTestRuns = 10                    # repeat self-test N times for confidence
    $SelfTestOnly = $false                # if true, exit after self-test without organizing
    $RunManifestPath = ''                 # optional path to write a run manifest JSON

    # Exclusions inside your canonical folder (safe defaults)
    $ExcludeRoots = @(
        'C:\Users\andrew\AppData',
        'C:\Users\andrew\OneDriveTemp',
        'C:\Users\andrew\.cache',
        'C:\Users\andrew\.nuget',
        'C:\Users\andrew\.vscode',
        'C:\Users\andrew\.git'
    )

    # =========================
    # Helpers
    # =========================
    function Ensure-Dir([string]$Path) {
        if (-not (Test-Path -LiteralPath $Path)) { New-Item -ItemType Directory -Path $Path -Force | Out-Null }
    }

    function Normalize-Root([string]$p) {
        if ([string]::IsNullOrWhiteSpace($p)) { return $null }
        $full = [System.IO.Path]::GetFullPath($p)
        if ($full.EndsWith('\')) { return $full }
        return $full + '\'
    }

    function Build-ExcludeList([string[]]$Roots) {
        $list = New-Object System.Collections.Generic.List[string]
        foreach ($r in $Roots) {
            try {
                $nr = Normalize-Root $r
                if ($nr) { $list.Add($nr) }
            } catch { }
        }
        return $list.ToArray()
    }

    function Is-ExcludedPath([string]$Path, [string[]]$ExcludeNorm) {
        if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
        $p = $Path
        if (-not $p.EndsWith('\')) { $p = $p + '\' }
        foreach ($ex in $ExcludeNorm) {
            if ($p.StartsWith($ex, [System.StringComparison]::OrdinalIgnoreCase)) { return $true }
        }
        return $false
    }

    function To-LongPath([string]$Path) {
        if ($Path -like '\\?\*') { return $Path }
        if ($Path -like '\\*') { return "\\?\UNC\" + $Path.TrimStart('\\') }
        return "\\?\" + $Path
    }

    function Get-StringMD5([string]$InputString) {
        $md5 = [System.Security.Cryptography.MD5]::Create()
        try {
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($InputString)
            $hashBytes = $md5.ComputeHash($bytes)
            return -join ($hashBytes | ForEach-Object { $_.ToString('x2') })
        } finally { $md5.Dispose() }
    }

    function Get-FileSHA256([string]$Path) {
        return (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash
    }

    function CopyOrMove-File([string]$Src, [string]$Dst, [bool]$Move) {
        try {
            if ($Move) { Move-Item -LiteralPath $Src -Destination $Dst -Force -ErrorAction Stop }
            else { Copy-Item -LiteralPath $Src -Destination $Dst -Force -ErrorAction Stop }
            return $true
        } catch {
            try {
                $srcLP = To-LongPath $Src
                $dstLP = To-LongPath $Dst
                if ($Move) { [System.IO.File]::Move($srcLP, $dstLP) }
                else { [System.IO.File]::Copy($srcLP, $dstLP, $true) }
                return $true
            } catch {
                return $false
            }
        }
    }

    function Get-FilesSafe([string]$Root, [bool]$SkipReparse, [string[]]$ExcludeNorm) {
        $rootFull = [System.IO.Path]::GetFullPath($Root)
        if (Is-ExcludedPath -Path $rootFull -ExcludeNorm $ExcludeNorm) { return }

        $rootDI = New-Object System.IO.DirectoryInfo($rootFull)
        $stack = New-Object System.Collections.Stack
        $stack.Push($rootDI)

        while ($stack.Count -gt 0) {
            $dir = $stack.Pop()
            try {
                if (Is-ExcludedPath -Path $dir.FullName -ExcludeNorm $ExcludeNorm) { continue }

                foreach ($sub in $dir.GetDirectories()) {
                    if ($SkipReparse -and (($sub.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) { continue }
                    if (Is-ExcludedPath -Path $sub.FullName -ExcludeNorm $ExcludeNorm) { continue }
                    $stack.Push($sub)
                }

                foreach ($file in $dir.GetFiles()) {
                    if ($SkipReparse -and (($file.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)) { continue }
                    $file
                }
            } catch {
                continue
            }
        }
    }

    function Remove-EmptyDirs([string[]]$Roots, [string[]]$ExcludeNorm) {
        foreach ($r in $Roots) {
            if (-not (Test-Path -LiteralPath $r)) { continue }
            $rFull = [System.IO.Path]::GetFullPath($r)
            if (Is-ExcludedPath -Path $rFull -ExcludeNorm $ExcludeNorm) { continue }

            $dirs = Get-ChildItem -LiteralPath $rFull -Directory -Recurse -Force -ErrorAction SilentlyContinue |
                Sort-Object { $_.FullName.Length } -Descending

            foreach ($d in $dirs) {
                try {
                    if (Is-ExcludedPath -Path $d.FullName -ExcludeNorm $ExcludeNorm) { continue }
                    $hasFiles = Get-ChildItem -LiteralPath $d.FullName -File -Force -ErrorAction SilentlyContinue | Select-Object -First 1
                    $hasDirs  = Get-ChildItem -LiteralPath $d.FullName -Directory -Force -ErrorAction SilentlyContinue | Select-Object -First 1
                    if (-not $hasFiles -and -not $hasDirs) {
                        Remove-Item -LiteralPath $d.FullName -Force -ErrorAction SilentlyContinue
                    }
                } catch { }
            }
        }
    }

    function Write-RowCsv([string]$CsvPath, [object]$Obj) {
        if (-not (Test-Path -LiteralPath $CsvPath)) {
            $Obj | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Encoding UTF8
        } else {
            $Obj | Export-Csv -LiteralPath $CsvPath -NoTypeInformation -Append -Encoding UTF8
        }
    }

    function Write-RowJsonl([string]$JsonlPath, [object]$Obj) {
        ($Obj | ConvertTo-Json -Compress) | Add-Content -LiteralPath $JsonlPath -Encoding UTF8
    }

    function Get-ExtBucket([System.IO.FileInfo]$File) {
        $ext = $File.Extension
        if ([string]::IsNullOrWhiteSpace($ext)) { return '_no_extension' }
        $clean = $ext.TrimStart('.').ToLowerInvariant()
        if ([string]::IsNullOrWhiteSpace($clean)) { return '_no_extension' }
        return $clean
    }

    function Get-UniqueDestName([System.IO.FileInfo]$File, [string]$SrcLeaf) {
        $pathHash = (Get-StringMD5 -InputString $File.FullName).Substring(0, 10)
        return "{0}__{1}__{2}" -f $SrcLeaf, $pathHash, $File.Name
    }

    function Invoke-OrganizeByExtension {
        param(
            [string[]]$Sources,
            [string]$DestRoot,
            [bool]$MoveMode,
            [bool]$DoDedupe,
            [string]$DupAction,
            [bool]$SkipReparse,
            [bool]$DoHash,
            [string[]]$ExcludeNorm
        )

        Ensure-Dir $DestRoot
        $logsDir = Join-Path $DestRoot '__LOGS'
        $dupsDir = Join-Path $DestRoot '__DUPLICATES'
        Ensure-Dir $logsDir
        if ($DoDedupe -and ($DupAction -eq 'Quarantine')) { Ensure-Dir $dupsDir }

        $ts = (Get-Date).ToString('yyyyMMdd_HHmmss')
        $csvLog = Join-Path $logsDir ("organize_by_ext_{0}.csv" -f $ts)
        $jsonl  = Join-Path $logsDir ("organize_by_ext_{0}.jsonl" -f $ts)

        $hashIndex = @{}   # sha256 -> primary dest path
        $counts = @{
            COPIED = 0; MOVED = 0; DUP_SKIPPED = 0; DUP_QUARANTINED = 0; FAILED = 0; SOURCE_MISSING = 0
        }

        foreach ($srcRoot in $Sources) {
            if ([string]::IsNullOrWhiteSpace($srcRoot)) { continue }
            if (-not (Test-Path -LiteralPath $srcRoot)) {
                $counts.SOURCE_MISSING++
                $row = [pscustomobject]@{
                    time_utc = (Get-Date).ToUniversalTime().ToString('o')
                    action   = 'SOURCE_MISSING'
                    source_root = $srcRoot
                    source   = $null
                    dest     = $null
                    ext      = $null
                    bytes    = $null
                    sha256   = $null
                    note     = $null
                    error    = $null
                }
                Write-RowCsv  $csvLog $row
                Write-RowJsonl $jsonl $row
                continue
            }

            $srcLeaf = Split-Path -Path $srcRoot -Leaf
            if ([string]::IsNullOrWhiteSpace($srcLeaf)) { $srcLeaf = 'andrew' }

            $files = @(Get-FilesSafe -Root $srcRoot -SkipReparse:$SkipReparse -ExcludeNorm $ExcludeNorm)

            foreach ($f in $files) {
                $extBucket = Get-ExtBucket $f
                $extDir = Join-Path $DestRoot $extBucket
                Ensure-Dir $extDir

                $destName = Get-UniqueDestName -File $f -SrcLeaf $srcLeaf
                $destPath = Join-Path $extDir $destName

                $sha = $null
                if ($DoHash -or $DoDedupe) {
                    try { $sha = Get-FileSHA256 -Path $f.FullName } catch { $sha = $null }
                }

                if ($DoDedupe -and $sha) {
                    if ($hashIndex.ContainsKey($sha)) {
                        if ($DupAction -eq 'Skip') {
                            $counts.DUP_SKIPPED++
                            $row = [pscustomobject]@{
                                time_utc = (Get-Date).ToUniversalTime().ToString('o')
                                action   = 'DUPLICATE_SKIPPED'
                                source_root = $srcRoot
                                source   = $f.FullName
                                dest     = $null
                                ext      = $extBucket
                                bytes    = $f.Length
                                sha256   = $sha
                                note     = "primary=" + $hashIndex[$sha]
                                error    = $null
                            }
                            Write-RowCsv  $csvLog $row
                            Write-RowJsonl $jsonl $row
                            continue
                        }

                        if ($DupAction -eq 'Quarantine') {
                            $qDir = Join-Path $dupsDir $extBucket
                            Ensure-Dir $qDir
                            $qPath = Join-Path $qDir $destName

                            $okQ = CopyOrMove-File -Src $f.FullName -Dst $qPath -Move:$MoveMode
                            if ($okQ) {
                                $counts.DUP_QUARANTINED++
                                $row = [pscustomobject]@{
                                    time_utc = (Get-Date).ToUniversalTime().ToString('o')
                                    action   = $(if ($MoveMode) { 'DUPLICATE_MOVED_TO_QUARANTINE' } else { 'DUPLICATE_COPIED_TO_QUARANTINE' })
                                    source_root = $srcRoot
                                    source   = $f.FullName
                                    dest     = $qPath
                                    ext      = $extBucket
                                    bytes    = $f.Length
                                    sha256   = $sha
                                    note     = "primary=" + $hashIndex[$sha]
                                    error    = $null
                                }
                                Write-RowCsv  $csvLog $row
                                Write-RowJsonl $jsonl $row
                                continue
                            } else {
                                $counts.FAILED++
                                $row = [pscustomobject]@{
                                    time_utc = (Get-Date).ToUniversalTime().ToString('o')
                                    action   = 'FAILED_DUP_QUARANTINE'
                                    source_root = $srcRoot
                                    source   = $f.FullName
                                    dest     = $qPath
                                    ext      = $extBucket
                                    bytes    = $f.Length
                                    sha256   = $sha
                                    note     = "primary=" + $hashIndex[$sha]
                                    error    = 'CopyOrMove failed'
                                }
                                Write-RowCsv  $csvLog $row
                                Write-RowJsonl $jsonl $row
                                continue
                            }
                        }
                    }
                }

                $ok = CopyOrMove-File -Src $f.FullName -Dst $destPath -Move:$MoveMode
                if ($ok) {
                    if ($MoveMode) { $counts.MOVED++ } else { $counts.COPIED++ }
                    if ($DoDedupe -and $sha -and (-not $hashIndex.ContainsKey($sha))) { $hashIndex[$sha] = $destPath }

                    $row = [pscustomobject]@{
                        time_utc = (Get-Date).ToUniversalTime().ToString('o')
                        action   = $(if ($MoveMode) { 'MOVED' } else { 'COPIED' })
                        source_root = $srcRoot
                        source   = $f.FullName
                        dest     = $destPath
                        ext      = $extBucket
                        bytes    = $f.Length
                        sha256   = $sha
                        note     = $null
                        error    = $null
                    }
                    Write-RowCsv  $csvLog $row
                    Write-RowJsonl $jsonl $row
                } else {
                    $counts.FAILED++
                    $row = [pscustomobject]@{
                        time_utc = (Get-Date).ToUniversalTime().ToString('o')
                        action   = 'FAILED'
                        source_root = $srcRoot
                        source   = $f.FullName
                        dest     = $destPath
                        ext      = $extBucket
                        bytes    = $f.Length
                        sha256   = $sha
                        note     = $null
                        error    = 'CopyOrMove failed'
                    }
                    Write-RowCsv  $csvLog $row
                    Write-RowJsonl $jsonl $row
                }
            }
        }

        $indexPath = Join-Path $logsDir ("index_by_ext_{0}.csv" -f $ts)
        $extDirs = Get-ChildItem -LiteralPath $DestRoot -Directory -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notin @('__LOGS','__DUPLICATES') }

        $idxRows = foreach ($d in $extDirs) {
            $files = @(Get-ChildItem -LiteralPath $d.FullName -File -Force -ErrorAction SilentlyContinue)
            $total = 0L
            foreach ($ff in $files) { $total += [int64]$ff.Length }
            [pscustomobject]@{
                ext_bucket  = $d.Name
                file_count  = $files.Count
                total_bytes = $total
                folder      = $d.FullName
            }
        }
        $idxRows | Sort-Object file_count -Descending | Export-Csv -LiteralPath $indexPath -NoTypeInformation -Encoding UTF8

        return [pscustomobject]@{
            DestinationRoot = $DestRoot
            CsvLog          = $csvLog
            JsonlLog        = $jsonl
            IndexByExt      = $indexPath
            Counts          = $counts
        }
    }

    function Invoke-SelfTest {
        $guid = [Guid]::NewGuid().ToString('N')
        $base = Join-Path $env:TEMP ("OrgByExt_SelfTest_{0}" -f $guid)
        $src1 = Join-Path $base 'src1'
        $src2 = Join-Path $base 'src2'
        $dst  = Join-Path $base 'dst'

        $exNorm = Build-ExcludeList -Roots @()

        try {
            Ensure-Dir $src1
            Ensure-Dir $src2
            Ensure-Dir $dst
            Ensure-Dir (Join-Path $src1 'nested')
            Ensure-Dir (Join-Path $src2 'nested')

            Set-Content -LiteralPath (Join-Path $src1 'a.txt') -Value 'same-content' -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $src2 'b.txt') -Value 'same-content' -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $src1 'c.pdf') -Value 'pdf-content'  -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $src2 'd')     -Value 'noext'        -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $src1 'nested\e.jpg') -Value 'img'  -Encoding UTF8
            Set-Content -LiteralPath (Join-Path $src2 'nested\f.jpg') -Value 'img2' -Encoding UTF8

            $r = Invoke-OrganizeByExtension -Sources @($src1,$src2) -DestRoot $dst -MoveMode:$false -DoDedupe:$true -DupAction:'Quarantine' -SkipReparse:$true -DoHash:$true -ExcludeNorm $exNorm

            if (-not (Test-Path -LiteralPath (Join-Path $dst 'txt'))) { throw 'SelfTest: missing txt folder' }
            if (-not (Test-Path -LiteralPath (Join-Path $dst 'pdf'))) { throw 'SelfTest: missing pdf folder' }
            if (-not (Test-Path -LiteralPath (Join-Path $dst 'jpg'))) { throw 'SelfTest: missing jpg folder' }
            if (-not (Test-Path -LiteralPath (Join-Path $dst '_no_extension'))) { throw 'SelfTest: missing _no_extension folder' }
            if (-not (Test-Path -LiteralPath (Join-Path (Join-Path $dst '__DUPLICATES') 'txt'))) { throw 'SelfTest: missing dup quarantine txt folder' }
            if (-not (Test-Path -LiteralPath $r.CsvLog)) { throw 'SelfTest: missing CSV log' }
            if (-not (Test-Path -LiteralPath $r.IndexByExt)) { throw 'SelfTest: missing index_by_ext CSV' }

            $txtPrimary = @(Get-ChildItem -LiteralPath (Join-Path $dst 'txt') -File -Force -ErrorAction SilentlyContinue)
            $txtDup = @(Get-ChildItem -LiteralPath (Join-Path (Join-Path $dst '__DUPLICATES') 'txt') -File -Force -ErrorAction SilentlyContinue)
            if ($txtPrimary.Count -ne 1) { throw "SelfTest: expected 1 primary txt file, got $($txtPrimary.Count)" }
            if ($txtDup.Count -ne 1) { throw "SelfTest: expected 1 quarantined dup txt file, got $($txtDup.Count)" }

            return $true
        } finally {
            try { Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue } catch { }
        }
    }

    # =========================
    # MAIN
    # =========================
    try {
        if ($Dedupe -and ($DedupeAction -notin @('Quarantine','Skip','KeepAll'))) {
            throw "Invalid DedupeAction. Use: Quarantine, Skip, KeepAll."
        }

        $excludeNorm = Build-ExcludeList -Roots $ExcludeRoots

        Write-Host "Organize start : $([DateTime]::Now.ToString('o'))"
        Write-Host "Source         : C:\Users\andrew"
        Write-Host "Mode           : $(if ($DoMove) { 'MOVE' } else { 'COPY' })"
        Write-Host "Destination    : $DestinationRoot"
        Write-Host "Group by ext   : YES"
        Write-Host "Dedupe         : $Dedupe"
        Write-Host "DedupeAction   : $DedupeAction"
        Write-Host "Skip reparse   : $SkipReparsePoints"
        Write-Host "Exclude roots  :"
        foreach ($x in $excludeNorm) { Write-Host "  - $x" }
        Write-Host ""

        if ($RunSelfTest) {
            for ($i = 1; $i -le $SelfTestRuns; $i++) {
                Write-Host "Self-test (temp sandbox) running... ($i/$SelfTestRuns)"
                $ok = Invoke-SelfTest
                if (-not $ok) { throw "Self-test returned failure." }
            }
            Write-Host "Self-test: PASS ($SelfTestRuns runs)"
            Write-Host ""
            if ($SelfTestOnly) {
                Write-Host "Self-test-only mode enabled. Exiting."
                return
            }
        }

        Ensure-Dir $DestinationRoot

        $result = Invoke-OrganizeByExtension `
            -Sources $SourceRoots `
            -DestRoot $DestinationRoot `
            -MoveMode:$DoMove `
            -DoDedupe:$Dedupe `
            -DupAction:$DedupeAction `
            -SkipReparse:$SkipReparsePoints `
            -DoHash:$ComputeSHA256 `
            -ExcludeNorm $excludeNorm

        if ($DoMove -and $RemoveEmptySourceFolders) {
            Write-Host "Removing empty folders under source roots (excluding protected roots)..."
            Remove-EmptyDirs -Roots $SourceRoots -ExcludeNorm $excludeNorm
        }

        if (-not [string]::IsNullOrWhiteSpace($RunManifestPath)) {
            $manifest = [pscustomobject]@{
                generated_utc     = (Get-Date).ToUniversalTime().ToString('o')
                sources           = $SourceRoots
                destination       = $DestinationRoot
                move_mode         = $DoMove
                dedupe            = $Dedupe
                dedupe_action     = if ($Dedupe) { $DedupeAction } else { 'OFF' }
                remove_empty_dirs = $RemoveEmptySourceFolders
                exclude_roots     = $excludeNorm
                logs              = [pscustomobject]@{
                    csv_log   = $result.CsvLog
                    jsonl_log = $result.JsonlLog
                    index_csv = $result.IndexByExt
                }
                counts            = $result.Counts
            }
            $manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $RunManifestPath -Encoding UTF8
        }

        Write-Host ""
        Write-Host "Organize complete: $([DateTime]::Now.ToString('o'))"
        Write-Host "Destination      : $($result.DestinationRoot)"
        Write-Host "Index (by ext)   : $($result.IndexByExt)"
        Write-Host "CSV Log          : $($result.CsvLog)"
        Write-Host "JSONL Log        : $($result.JsonlLog)"
        Write-Host "Counts           : COPIED=$($result.Counts.COPIED) MOVED=$($result.Counts.MOVED) DUP_SKIPPED=$($result.Counts.DUP_SKIPPED) DUP_QUARANTINED=$($result.Counts.DUP_QUARANTINED) FAILED=$($result.Counts.FAILED) SOURCE_MISSING=$($result.Counts.SOURCE_MISSING)"
    }
    catch {
        Write-Host ""
        Write-Host "FATAL: $($_.Exception.Message)"
        Write-Host "STACK: $($_.ScriptStackTrace)"
        throw
    }
}
