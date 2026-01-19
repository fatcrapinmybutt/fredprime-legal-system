# Troubleshooting Guide - DriveOrganizerPro

## Common Issues and Solutions

### GUI Issues

#### GUI Won't Start

**Symptoms**: Error when running `driveorganizerpro-gui` or double-clicking launcher

**Solutions**:

1. **Check Python installation**

   ```bash
   python --version
   # Should be 3.9 or higher
   ```

2. **Check tkinter availability**

   ```bash
   python -m tkinter
   # Should open a small test window
   ```

3. **Reinstall tkinter**

   ```bash
   # Ubuntu/Debian
   sudo apt install python3-tk

   # Fedora
   sudo dnf install python3-tkinter
   ```

4. **Use command line mode instead**

   ```python
   from drive_organizer_pro.core.organizer_engine import OrganizerEngine
   engine = OrganizerEngine()
   ```

#### GUI Freezes During Operation

**Symptoms**: Window becomes unresponsive

**Cause**: Processing very large numbers of files

**Solutions**:
- Be patient - operations run in background thread
- Check log output for progress
- For very large operations, use command line mode

### File Operation Issues

#### Permission Denied Errors

**Symptoms**: "Permission denied" in log output

**Solutions**:

1. **Run as administrator** (Windows)
   - Right-click launcher
   - Select "Run as administrator"

2. **Check file permissions** (Linux/macOS)

   ```bash
   chmod +x /path/to/file
   ```

3. **Close programs using files**
   - Close any programs that may have files open
   - Check for running backup software

#### Files Not Moving

**Symptoms**: Organization completes but files haven't moved

**Solutions**:

1. **Check if DRY RUN is enabled**
   - Look for "DRY RUN" in log output
   - Uncheck "DRY RUN" option

2. **Verify source path**
   - Ensure path exists
   - Check for typos in path

3. **Check bucket configuration**
   - Verify file extensions are in bucket definitions
   - Files with unrecognized extensions go to "Miscellaneous"

#### Name Collision Errors

**Symptoms**: Errors about existing files

**Cause**: Files with same name in destination

**Solution**: System automatically renames with counter (e.g., "file (1).txt")
- No action needed from user
- Check renamed files in destination

### Duplicate Detection Issues

#### Duplicates Not Detected

**Symptoms**: Expected duplicates not quarantined

**Solutions**:

1. **Enable duplicate detection**
   - Check "Detect and quarantine duplicates"

2. **Large files take time**
   - Hashing large files is slow
   - Monitor log for hash calculation messages

3. **Different file formats**
   - Only exact binary duplicates are detected
   - Similar content in different formats won't match

#### False Duplicate Detection

**Symptoms**: Different files marked as duplicates

**Cause**: Extremely rare hash collision

**Solution**:
- Check `_Duplicates` folder
- Manually restore false positives
- Report issue on GitHub

### Performance Issues

#### Slow Organization

**Symptoms**: Takes very long to process files

**Solutions**:

1. **Reduce worker threads**

   ```python
   stats = engine.organize_drive(
       source_path=path,
       max_workers=2  # Default is 8
   )
   ```

2. **Disable duplicate detection**
   - Hashing is CPU-intensive
   - Uncheck "Detect duplicates" for faster operation

3. **Process smaller directories**
   - Organize subdirectories individually
   - Combine results manually

#### High Memory Usage

**Symptoms**: System runs out of memory

**Cause**: Processing very large number of files

**Solutions**:

1. **Process in batches**
   - Organize one folder at a time

2. **Increase system RAM**

3. **Close other applications**

### Revert Issues

#### Revert Doesn't Work

**Symptoms**: Files not moved back after revert

**Solutions**:

1. **Check session exists**
   - Revert only works if backup session saved
   - Look for `.driveorganizerpro/backups/` folder

2. **Files moved manually**
   - Revert requires files still at organized location
   - Can't revert if files moved elsewhere

3. **Session corrupted**
   - Check backup JSON file is valid
   - Manually restore from JSON if needed

### Error Messages

#### "No session found to revert"

**Cause**: No backup session available

**Solution**: Organization must complete successfully to create backup session

#### "Failed to calculate MD5"

**Cause**: File is locked or inaccessible

**Solution**:
- Close programs using the file
- Check file permissions
- File will be skipped

#### "OSError: [Errno 36] File name too long"

**Cause**: Path exceeds OS limits

**Solution**:
- Shorten file names before organizing
- Use shorter bucket names
- Enable long path support (Windows):
  - Run: `reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1`

## Getting More Help

### Before Asking for Help

1. Check this troubleshooting guide
2. Review the [FAQ](FAQ.md)
3. Check [User Guide](USER_GUIDE.md)
4. Search [GitHub Issues](https://github.com/fatcrapinmybutt/fredprime-legal-system/issues)

### Reporting Issues

When reporting issues, include:

1. **Error message** (exact text)
2. **Log output** (from GUI or console)
3. **System information**:
   - OS and version
   - Python version
   - DriveOrganizerPro version
4. **Steps to reproduce**
5. **Expected vs actual behavior**

### Log Files

Logs are stored in:
- GUI mode: Check log panel
- Command line: Default to console

Enable file logging:

```python
from drive_organizer_pro.utils.logger import setup_logger
logger = setup_logger(log_file=Path("organizer.log"))
```

---

© 2026 MBP LLC. All rights reserved. Powered by Pork™
