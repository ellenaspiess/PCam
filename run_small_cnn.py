# run_small_cnn.py
from pathlib import Path
import sys

# Projekt-Root ermitteln und zu sys.path hinzufügen
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.training.train_small_cnn import train


if __name__ == "__main__":
    # Hier kannst du später easy Parameter anpassen
    train(
        data_root="data/raw",
        num_epochs=5,
        batch_size=64,
        lr=1e-3,
        dropout_p=0.1,
    )
