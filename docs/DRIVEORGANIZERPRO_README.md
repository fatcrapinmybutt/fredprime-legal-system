# 🐷 DriveOrganizerPro - MBP LLC

```text
████████████████████████████████████████████████████████████████████
██                                                                ██
██   ███╗   ███╗██████╗ ██████╗     ██╗     ██╗      ██████╗    ██
██   ████╗ ████║██╔══██╗██╔══██╗    ██║     ██║     ██╔════╝    ██
██   ██╔████╔██║██████╔╝██████╔╝    ██║     ██║     ██║         ██
██   ██║╚██╔╝██║██╔══██╗██╔═══╝     ██║     ██║     ██║         ██
██   ██║ ╚═╝ ██║██████╔╝██║         ███████╗███████╗╚██████╗    ██
██   ╚═╝     ╚═╝╚═════╝ ╚═╝         ╚══════╝╚══════╝ ╚═════╝    ██
██                                                                ██
██              MAXIMUM BUSINESS PERFORMANCE                      ██
██                 "Powered by Pork™"                             ██
██                                                                ██
████████████████████████████████████████████████████████████████████
```

## Level 9999 Drive Organizer - The Ultimate File Management System

[![Version](https://img.shields.io/badge/version-1.0.0-gold)](https://github.com/fatcrapinmybutt/fredprime-legal-system)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![MBP LLC](https://img.shields.io/badge/Powered%20by-Pork™-ff69b4)](https://github.com/fatcrapinmybutt)

**Transform chaos into order with the power of pork!** DriveOrganizerPro is a professional-grade file organization system
that automatically categorizes, de-nests, and manages files across your drives with intelligence and precision.

---

## 🔥 Features

### Core Capabilities

- **🎯 Smart Bucketization** - Automatically categorizes files into 15 intelligent buckets
- **🗂️ Sub-Bucket Organization** - Creates 8 specialized sub-folders for targeted workflows
- **🔍 Duplicate Detection** - MD5/SHA256 hashing to find and quarantine duplicates
- **⏮️ Full Revert System** - JSON-based backup with complete rollback capability
- **🚀 Multi-threaded** - Lightning-fast processing with concurrent operations
- **🌑 Dark Theme GUI** - Professional, easy-on-the-eyes interface
- **💾 Persistent State** - Resume operations across sessions
- **🧹 Empty Folder Cleanup** - Automatically removes empty directories

### Advanced Features

- **Keyword Detection** - Smart categorization for legal, technical, and specialized files
- **Dry Run Mode** - Preview changes before committing
- **Progress Tracking** - Real-time updates with detailed statistics
- **Error Logging** - Comprehensive logging with rotating file handlers
- **Safe Operations** - Atomic file moves with collision detection
- **Cross-Platform** - Windows, macOS, and Linux support

---

## 📦 Installation

### Quick Install (Windows)

```batch
# Clone the repository
git clone https://github.com/fatcrapinmybutt/fredprime-legal-system.git
cd fredprime-legal-system

# Run installer
scripts\install.bat
```

### Manual Install

```bash
# Install with pip
pip install -e .

# Or install from source
python setup.py install
```

### Requirements

- Python 3.9 or higher
- tkinter (included with Python)
- No external dependencies!

---

## 🚀 Quick Start

### GUI Mode (Recommended)

```bash
# Launch the GUI
driveorganizerpro-gui

# Or use the launcher
python launcher.pyw
```

### Command Line Mode

```python
from pathlib import Path
from drive_organizer_pro.core.organizer_engine import OrganizerEngine

# Create engine
engine = OrganizerEngine()

# Organize a drive
stats = engine.organize_drive(
    source_path=Path("E:/"),
    dry_run=True,  # Preview first!
    handle_duplicates=True,
    create_sub_buckets=True
)

print(f"Processed {stats['files_moved']} files!")
```

---

## 📊 Bucket Structure

DriveOrganizerPro organizes files into **15 intelligent buckets**:

| Bucket | Extensions |
|--------|-----------|
| **Documents** | .pdf, .doc, .docx, .txt, .rtf, .odt |
| **Spreadsheets** | .xls, .xlsx, .csv, .ods |
| **Images** | .jpg, .jpeg, .png, .gif, .bmp, .svg, .webp, .tiff, .heic |
| **Videos** | .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm |
| **Audio** | .mp3, .wav, .flac, .aac, .ogg, .m4a, .wma |
| **Archives** | .zip, .rar, .7z, .tar, .gz, .tgz, .bz2, .iso |
| **Code** | .py, .js, .html, .css, .cpp, .java, .c, .cs, .php, .rb, .go, .ts |
| **Databases** | .db, .sqlite, .mdb, .sql |
| **Executables** | .exe, .msi, .bat, .ps1, .sh, .cmd |
| **Legal_Court** | (Smart detection) |
| **Neo4j_Data** | .cypher, .cql, .graphml |
| **Presentations** | .ppt, .pptx, .key, .odp |
| **Books_PDFs** | .epub, .mobi, .azw |
| **Email** | .eml, .msg, .pst |
| **Miscellaneous** | Everything else |

### Sub-Buckets (Within Each Bucket)

1. **Meek1_Housing_ShadyOaks** - Housing and landlord-related files
2. **Meek2_Custody** - Custody and parenting time documents
3. **Meek3_PPO** - Protective order related files
4. **Meek4_Court_Violations** - Court violations and judicial matters
5. **LitigationOS** - Litigation system files
6. **Neo4j_Graphs** - Graph database files
7. **Michigan_Court_Authorities** - Michigan legal authorities
8. **Studying_Materials** - Educational and study content

---

## 🎨 GUI Screenshot (ASCII Art)

```text
┌─────────────────────────────────────────────────────────────────┐
│  🐷 MBP LLC DriveOrganizerPro - Level 9999 Edition              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Source Directory:                                              │
│  [C:\Users\Documents                            ] [Browse...]   │
│                                                                 │
│  Options:                                                       │
│  ☑ DRY RUN (Preview Only - RECOMMENDED FIRST TIME)            │
│  ☑ Remove empty directories after organization                │
│  ☑ Detect and quarantine duplicate files                      │
│  ☑ Create sub-buckets (Meek1-4, LitigationOS, etc.)           │
│                                                                 │
│  [🚀 ORGANIZE DRIVES]  [⚠️ REVERT CHANGES]  [Clear Log]        │
│                                                                 │
│  Progress:                                                      │
│  ████████████████████████░░░░░░░░  80% (1,234/1,500 files)    │
│                                                                 │
│  Log Output:                                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │ [INFO] Starting organization (DRY RUN)...                 │ │
│  │ [INFO] Discovered 1,500 files to process                 │ │
│  │ [INFO] Moving: document.pdf -> Documents/Meek2_Custody    │ │
│  │ [SUCCESS] Organization complete!                          │ │
│  │ [INFO] Files processed: 1,500                             │ │
│  │ [INFO] Files moved: 1,234                                 │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                 │
│  Status: Processing complete - Ready                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Architecture

### Component Overview

```text
DriveOrganizerPro
├── Core Engine
│   ├── OrganizerEngine    - Main orchestration
│   ├── FileAnalyzer       - Classification logic
│   ├── DuplicateHandler   - Hash-based deduplication
│   ├── BackupManager      - Revert system
│   └── SubBucketManager   - Keyword detection
├── Configuration
│   ├── ConfigManager      - Settings management
│   ├── default_buckets    - Bucket definitions
│   └── sub_buckets        - Keyword mappings
├── GUI System
│   ├── MainWindow         - Application window
│   ├── Components         - Reusable widgets
│   ├── Themes             - MBP dark theme
│   └── SplashScreen       - Startup screen
└── Utilities
    ├── Logger             - Professional logging
    ├── FileUtils          - Safe file operations
    ├── HashUtils          - MD5/SHA256 hashing
    └── PathUtils          - Cross-platform paths
```

---

## 📖 Usage Examples

### Example 1: Basic Organization

```python
from pathlib import Path
from drive_organizer_pro.core.organizer_engine import OrganizerEngine

engine = OrganizerEngine()

# Organize a single directory
stats = engine.organize_drive(
    source_path=Path("C:/MyFiles"),
    dry_run=False,
    remove_empty=True
)
```

### Example 2: With Progress Callback

```python
def my_progress(current, total, status):
    print(f"{current}/{total}: {status}")

stats = engine.organize_drive(
    source_path=Path("E:/"),
    progress_callback=my_progress,
    max_workers=8
)
```

### Example 3: Revert Operation

```python
# Revert the last organization
count = engine.revert_last_organization(dry_run=False)
print(f"Reverted {count} files")
```

---

## 🔧 Configuration

### Custom Buckets

Create `custom_buckets.json`:

```json
{
  "MyCustomBucket": [".custom", ".special"],
  "VideoProjects": [".prproj", ".aep", ".fcp"]
}
```

Load custom configuration:

```python
from drive_organizer_pro.config.config_manager import ConfigManager

config = ConfigManager()
config.load_buckets(Path("custom_buckets.json"))
```

### Custom Sub-Buckets

Modify `sub_buckets.json` to add your own keyword mappings:

```json
{
  "sub_buckets": ["MyProject", "ClientWork"],
  "keyword_mappings": {
    "MyProject": ["project", "myapp"],
    "ClientWork": ["client", "invoice"]
  }
}
```

---

## 🧪 Testing

```bash
# Run test suite
pytest tests/ -v --cov=src/drive_organizer_pro

# Or use the batch file (Windows)
scripts\run_tests.bat
```

---

## 📦 Building Executable

```bash
# Build standalone .exe (Windows)
scripts\build_exe.bat

# Output: dist/DriveOrganizerPro-MBP-LLC.exe
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🐷 About MBP LLC

**Maximum Business Performance, LLC** - Delivering excellence through innovation.

*"Powered by Pork™"* - Our commitment to quality and performance.

- **Company**: MBP LLC (Maximum Business Performance, LLC)
- **Tagline**: Powered by Pork™
- **Contact**: contact@mbpllc.example
- **Copyright**: © 2026 MBP LLC. All rights reserved.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/fatcrapinmybutt/fredprime-legal-system/issues)
- **Documentation**: [User Guide](docs/USER_GUIDE.md)
- **FAQ**: [Frequently Asked Questions](docs/FAQ.md)

---

## 🏆 Acknowledgments

- Built with Python and tkinter
- Inspired by the need for better file organization
- Powered by maximum business performance

---

<div align="center">

**🐷 Made with maximum business performance by MBP LLC 🐷**

**"Powered by Pork™"**

[⭐ Star us on GitHub](https://github.com/fatcrapinmybutt/fredprime-legal-system) | 
[🐛 Report Bug](https://github.com/fatcrapinmybutt/fredprime-legal-system/issues) | 
[✨ Request Feature](https://github.com/fatcrapinmybutt/fredprime-legal-system/issues)

</div>
