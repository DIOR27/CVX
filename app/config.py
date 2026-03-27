import os

from dotenv import load_dotenv

load_dotenv()

OLLAMA_URL = os.getenv(
    "OLLAMA_URL", "https://pruebask8s.ucuenca.edu.ec/qwen/api/generate"
)
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_CONNECT_TIMEOUT = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", "30"))
OLLAMA_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "45"))
OLLAMA_WRITE_TIMEOUT = float(os.getenv("OLLAMA_WRITE_TIMEOUT", "30"))
OLLAMA_EXTRACTION_READ_TIMEOUT = float(
    os.getenv("OLLAMA_EXTRACTION_READ_TIMEOUT", "25")
)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
VERIFY_MAX_TOKENS = int(os.getenv("VERIFY_MAX_TOKENS", "5"))
EXTRACTION_MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "1200"))
EXPERIENCE_SECTION_MAX_CHARS = int(os.getenv("EXPERIENCE_SECTION_MAX_CHARS", "3000"))
PDF_LEFT_CROP_RATIO = float(os.getenv("PDF_LEFT_CROP_RATIO", "0.68"))
PDF_RIGHT_CROP_START_RATIO = float(os.getenv("PDF_RIGHT_CROP_START_RATIO", "0.30"))
ENABLE_REMOTE_EXTRACTION = (
    os.getenv("ENABLE_REMOTE_EXTRACTION", "true").strip().lower() == "true"
)

APP_TITLE = "CV Analyzer API"
APP_VERSION = "5.1.0"
