import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 50
HIDDEN_DIM = 32
LEARNING_RATE = 0.01
EPOCHS = 80
PRINT_EVERY = 10
PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
LOCAL_GLOVE_DIR = PROJECT_ROOT / "data_store" / "glove"
LOCAL_GLOVE_FILE = LOCAL_GLOVE_DIR / "glove.6B.50d.txt"
RUNS_DIR = PROJECT_ROOT / "runs"
