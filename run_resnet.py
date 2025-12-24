# run_resnet.py (in project root, same level as run_small_cnn.py)
from src.training.train_resnet_pcam import train_resnet

if __name__ == "__main__":
    train_resnet(
        data_root="data/raw",
        num_epochs=2,      # start small
        batch_size=64,
        lr=1e-4,
        tl_mode="frozen",  # first run: frozen; second run: "partial"
    )
