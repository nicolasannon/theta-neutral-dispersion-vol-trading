import logging
from pathlib import Path

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)


log_file = output_dir / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__)
