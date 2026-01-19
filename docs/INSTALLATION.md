# Installation Guide - DriveOrganizerPro

## System Requirements

- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum (8GB recommended for large drives)
- **Disk Space**: 100MB for installation + space for file operations

## Installation Methods

### Method 1: Quick Install (Windows)

1. **Download the repository**

   ```batch
   git clone https://github.com/fatcrapinmybutt/fredprime-legal-system.git
   cd fredprime-legal-system
   ```

2. **Run the installer**

   ```batch
   scripts\install.bat
   ```

3. **Launch the application**

   ```batch
   driveorganizerpro-gui
   ```

### Method 2: pip Install

```bash
# Install from source
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

### Method 3: Standalone Executable (Windows)

1. **Build the executable**

   ```batch
   scripts\build_exe.bat
   ```

2. **Run the executable**

   ```batch
   dist\DriveOrganizerPro-MBP-LLC.exe
   ```

## Python Installation

### Windows

1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run installer and check "Add Python to PATH"
3. Verify installation:

   ```batch
   python --version
   ```

### macOS

```bash
# Using Homebrew
brew install python@3.11

# Verify
python3 --version
```

### Linux

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3-pip python3-tk

# Fedora
sudo dnf install python3.11 python3-pip python3-tkinter

# Verify
python3 --version
```

## Verifying Installation

```bash
# Check if module is importable
python -c "from drive_organizer_pro import __version__; print(__version__)"

# Should output: 1.0.0

# Test GUI launch
driveorganizerpro-gui
```

## Troubleshooting Installation

### Issue: "python not found"

**Solution**: Add Python to your system PATH

- Windows: Re-run Python installer, check "Add to PATH"
- macOS/Linux: Add to `.bashrc` or `.zshrc`:

  ```bash
  export PATH="/usr/local/bin/python3:$PATH"
  ```

### Issue: "tkinter not found"

**Solution**: Install tkinter

```bash
# Ubuntu/Debian
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# macOS (usually included)
brew install python-tk
```

### Issue: "pip install fails"

**Solution**: Upgrade pip and try again

```bash
python -m pip install --upgrade pip
pip install -e .
```

## Uninstallation

```bash
# Uninstall package
pip uninstall driveorganizerpro-mbpllc

# Remove configuration (optional)
rm -rf ~/.driveorganizerpro
```

## Next Steps

After installation:

1. Read the [User Guide](USER_GUIDE.md)
2. Review [Best Practices](USER_GUIDE.md#best-practices)
3. Try a dry run on a small directory

---

© 2026 MBP LLC. All rights reserved. Powered by Pork™
