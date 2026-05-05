from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJ_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MLRUNS_DIR = PROJ_ROOT / "mlruns"
MLFLOW_TRACKING_URI = MLRUNS_DIR.as_uri()

RANDOM_SEED = 12

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
