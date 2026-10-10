"""Central configuration for the shelf monitoring pipeline."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", PROJECT_ROOT / "output"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", PROJECT_ROOT / "cache"))
PLANOGRAM_PATH = Path(os.getenv("PLANOGRAM_PATH", PROJECT_ROOT / "planogram.json"))

# VLM Settings
VLM_PROVIDER = os.getenv("VLM_PROVIDER", "gemini").lower()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Image Processing
CROP_SIZE = (448, 448)          # Standard perspective crop dimensions (W, H)
JPEG_QUALITY = 90

# Inference & Uncertainty Policy
MAX_RETRIES = 3                 # Retry count for transient API drops
CONFIDENCE_THRESHOLD = 0.65     # Below this, fallback to conservative safe values
STOCK_BIAS = -0.1               # Asymmetric penalty protection (avoid overestimating stock)
