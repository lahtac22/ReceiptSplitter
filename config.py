from pathlib import Path

# --------------------------------------------------
# Directories
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
DEBUG_DIR = BASE_DIR / "debug"

# --------------------------------------------------
# PDF settings
# --------------------------------------------------

PDF_DPI = 300

# --------------------------------------------------
# Debugging
# --------------------------------------------------

DEBUG = True