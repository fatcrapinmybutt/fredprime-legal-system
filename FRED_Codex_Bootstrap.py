import os

"""Bootstrap script for FRED Prime Omnicourse.

This script clones the repository, downloads the Stage2 deployment
archive, extracts it, then runs the setup and GUI launch commands.
"""

REPO_URL = "https://github.com/fatcrapinmybutt/fredprime-legal-system.git"
ARCHIVE_URL = "https://sandbox.openai.com/mnt/data/FRED_STAGE2_FULL_DEPLOY.zip"
ARCHIVE_NAME = "FRED_STAGE2_FULL_DEPLOY.zip"

os.system(f"git clone {REPO_URL}")
os.chdir("fredprime-legal-system")
os.system(f"curl -O {ARCHIVE_URL}")
os.system(f"unzip -o {ARCHIVE_NAME} -d .")
os.system("bash SovereignSetup.sh && python3 gui/gui_launcher.py")
