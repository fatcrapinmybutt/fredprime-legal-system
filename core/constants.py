"""System-wide constants for FRED Supreme Litigation OS.

Defines standard values, paths, and configuration constants.
"""

from pathlib import Path

# Version information
VERSION = "1.0.0"
SYSTEM_NAME = "FRED Supreme Litigation OS"

# File paths
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / "state"
DATA_DIR = PROJECT_ROOT / "data"

# Evidence types
EVIDENCE_TYPES = [
    "documentary",
    "testimonial",
    "demonstrative",
    "physical",
    "digital",
]

# Document types
DOCUMENT_TYPES = [
    "motion",
    "affidavit",
    "exhibit",
    "order",
    "complaint",
    "response",
    "brief",
    "transcript",
    "correspondence",
]

# Case types
CASE_TYPES = [
    "custody",
    "ppo",
    "divorce",
    "civil",
    "criminal",
    "probate",
]

# Michigan Court Rules
MCR_COMPLIANCE_RULES = [
    "MCR 2.113",  # Form and signing of documents
    "MCR 2.114",  # Signatures, verifications
    "MCR 2.119",  # Motion practice
    "MCR 3.206",  # Pleadings
]

# Exhibit labeling (A-Z, AA-ZZ, etc.)
EXHIBIT_LABELS_SINGLE = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
EXHIBIT_LABELS_DOUBLE = [
    f"{chr(i)}{chr(j)}"
    for i in range(ord('A'), ord('Z') + 1)
    for j in range(ord('A'), ord('Z') + 1)
]

# File extensions
SUPPORTED_DOCUMENT_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".rtf",
]

SUPPORTED_IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".bmp",
]

SUPPORTED_VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
]

# System limits
MAX_FILE_SIZE_MB = 500
MAX_EVIDENCE_ITEMS = 10000
MAX_EXHIBIT_COUNT = 676  # A-Z (26) + AA-ZZ (676)

# Timeouts (seconds)
DEFAULT_TIMEOUT = 300
NETWORK_TIMEOUT = 30
DATABASE_TIMEOUT = 60

# Logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL = "INFO"

# Quality thresholds
MIN_CONFIDENCE_SCORE = 0.7
MIN_EVIDENCE_QUALITY = 0.6
MIN_DOCUMENT_COMPLETENESS = 0.8
