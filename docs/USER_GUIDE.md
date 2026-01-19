# User Guide - DriveOrganizerPro

## Table of Contents

- [Getting Started](#getting-started)
- [Using the GUI](#using-the-gui)
- [Command Line Usage](#command-line-usage)
- [Configuration](#configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### First Time Setup

1. **Install DriveOrganizerPro**

   ```bash
   pip install -e .
   ```

2. **Launch the Application**

   ```bash
   driveorganizerpro-gui
   ```

3. **Run a DRY RUN First**
   - Always check "DRY RUN" on your first use
   - This previews what changes will be made without actually moving files

## Using the GUI

### Main Window Components

#### 1. Source Directory Selection

- Click **Browse** to select the directory you want to organize
- You can manually type a path in the text field
- Supports both drive letters (E:, F:) and full paths

#### 2. Options Panel

**DRY RUN (Preview Only)**
- When checked, shows what would happen without making changes
- **ALWAYS recommended for first-time use**
- Review the log output before running for real

**Remove empty directories**
- Cleans up empty folders after organization
- Helps maintain a tidy file structure

**Detect and quarantine duplicates**
- Uses MD5 hashing to find duplicate files
- Moves duplicates to `_Duplicates` folder
- Keeps the oldest file as the original

**Create sub-buckets**
- Creates specialized sub-folders within each bucket
- Based on keyword detection (Meek1-4, LitigationOS, etc.)

#### 3. Action Buttons

**🚀 ORGANIZE DRIVES**
- Starts the organization process
- Disabled during operation

**⚠️ REVERT CHANGES**
- Undoes the last organization
- Moves files back to original locations

**Clear Log**
- Clears the log output panel

#### 4. Progress Panel

- Shows real-time progress bar
- Displays files processed count
- Shows current operation status

#### 5. Log Output

- Color-coded messages:
  - **Green**: Success messages
  - **Orange**: Warnings
  - **Red**: Errors
  - **White**: Info messages

### Step-by-Step Workflow

#### Step 1: Dry Run

1. Select your source directory
2. Check "DRY RUN"
3. Configure other options as desired
4. Click "ORGANIZE DRIVES"
5. Review the log output carefully

#### Step 2: Real Organization

1. If dry run looks good, uncheck "DRY RUN"
2. Click "ORGANIZE DRIVES" again
3. Confirm when prompted
4. Wait for completion

#### Step 3: Verify Results

1. Navigate to your source directory
2. Check the "Organized" folder
3. Verify files are in correct buckets
4. If needed, use "REVERT CHANGES"

## Command Line Usage

### Basic Organization

```python
from pathlib import Path
from drive_organizer_pro.core.organizer_engine import OrganizerEngine

# Create engine
engine = OrganizerEngine()

# Organize
stats = engine.organize_drive(
    source_path=Path("E:/"),
    dry_run=True
)

print(f"Moved {stats['files_moved']} files")
```

### With Custom Options

```python
stats = engine.organize_drive(
    source_path=Path("E:/Documents"),
    output_path=Path("E:/Organized"),
    dry_run=False,
    remove_empty=True,
    handle_duplicates=True,
    create_sub_buckets=True,
    max_workers=8
)
```

### Progress Callback

```python
def show_progress(current, total, status):
    percentage = (current / total) * 100
    print(f"{percentage:.1f}% - {status}")

stats = engine.organize_drive(
    source_path=Path("E:/"),
    progress_callback=show_progress
)
```

## Configuration

### Custom Buckets

Create a custom bucket configuration:

```json
{
  "MyBucket": [".custom", ".special"],
  "WorkFiles": [".work", ".project"]
}
```

Load it:

```python
from drive_organizer_pro.config.config_manager import ConfigManager

config = ConfigManager()
config.load_buckets(Path("my_buckets.json"))
```

### Custom Sub-Buckets

Modify keyword mappings:

```json
{
  "sub_buckets": ["Project1", "Project2"],
  "keyword_mappings": {
    "Project1": ["proj1", "alpha"],
    "Project2": ["proj2", "beta"]
  }
}
```

## Best Practices

### Before Organizing

1. **Backup Important Data**
   - Always have a backup before major file operations
   - Use the built-in revert system as a safety net

2. **Start with Dry Run**
   - Preview changes first
   - Verify bucket assignments
   - Check for unexpected moves

3. **Start Small**
   - Test on a small directory first
   - Verify results before processing large drives

### During Organization

1. **Monitor Progress**
   - Watch the log output for errors
   - Check file counts match expectations

2. **Don't Interrupt**
   - Let the process complete
   - Interrupting may leave files in inconsistent state

### After Organization

1. **Verify Results**
   - Check key files are where expected
   - Verify important files weren't missed

2. **Clean Up**
   - Empty directories are automatically removed
   - Check `_Duplicates` folder for any false positives

## Troubleshooting

### Common Issues

**"No files were moved"**
- Check if files exist in source directory
- Verify path is correct
- Ensure files aren't already organized

**"Permission denied" errors**
- Run as administrator (Windows)
- Check file/folder permissions
- Close programs using the files

**"Duplicate detection not working"**
- Ensure "Detect duplicates" is checked
- Large files take longer to hash
- Check log for hash calculation errors

**GUI won't start**
- Verify Python 3.9+ is installed
- Check tkinter is available: `python -m tkinter`
- Try command line mode

### Getting Help

1. Check the [FAQ](FAQ.md)
2. Review [Troubleshooting Guide](TROUBLESHOOTING.md)
3. Open an issue on GitHub

---

© 2026 MBP LLC. All rights reserved. Powered by Pork™
