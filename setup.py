"""
Setup script for DriveOrganizerPro - MBP LLC.

© 2026 MBP LLC. All rights reserved.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="driveorganizerpro-mbpllc",
    version="1.0.0",
    description="Level 9999 Drive Organizer - MBP LLC Edition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="MBP LLC",
    author_email="contact@mbpllc.example",
    url="https://github.com/fatcrapinmybutt/fredprime-legal-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "drive_organizer_pro": [
            "config/*.json",
        ],
    },
    python_requires=">=3.9",
    install_requires=[
        # No external dependencies - uses Python stdlib
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pyinstaller>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "driveorganizerpro=drive_organizer_pro.gui.main_window:main",
        ],
        "gui_scripts": [
            "driveorganizerpro-gui=drive_organizer_pro.gui.main_window:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: System :: Filesystems",
        "Topic :: Utilities",
    ],
    keywords="file organization, drive organizer, file manager, mbp llc",
    project_urls={
        "Bug Reports": "https://github.com/fatcrapinmybutt/fredprime-legal-system/issues",
        "Source": "https://github.com/fatcrapinmybutt/fredprime-legal-system",
    },
)
