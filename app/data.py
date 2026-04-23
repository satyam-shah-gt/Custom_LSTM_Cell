from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import kagglehub
import torch

from .config import EMBED_DIM, LOCAL_GLOVE_DIR, LOCAL_GLOVE_FILE


DATASET: List[Tuple[str, int]] = [
    ("i love this movie", 1),
    ("this film was amazing", 1),
    ("i really liked this film", 1),
    ("this movie was fantastic", 1),
    ("the acting was good", 1),
    ("the story was wonderful", 1),
    ("i enjoyed this movie a lot", 1),
    ("this was a great film", 1),
    ("the movie was excellent", 1),
    ("i am happy with this film", 1),
    ("i hate this movie", 0),
    ("this film was terrible", 0),
    ("i dislike this movie", 0),
    ("this movie was awful", 0),
    ("the acting was bad", 0),
    ("the story was horrible", 0),
    ("i regret watching this movie", 0),
    ("this was a boring film", 0),
    ("the movie was disappointing", 0),
    ("i am unhappy with this film", 0),
    ("i love this movie but the ending was terrible", 0),
    ("the story was good but the acting was bad", 0),
    ("the film started well but became awful", 0),
    ("i liked the first half but hated the ending", 0),
    ("i hate this movie but the ending was amazing", 1),
    ("the acting was bad but the story was great", 1),
    ("the film was boring but the climax was fantastic", 1),
    ("i disliked most of it but the ending was wonderful", 1),
]


def tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def find_glove_file(root: str | Path) -> Path:
    root = Path(root)
    candidates = list(root.rglob("glove.6B.50d.txt"))
    if not candidates:
        raise FileNotFoundError(f"Could not find glove.6B.50d.txt under: {root}")
    return candidates[0]


def download_glove() -> Path:
    LOCAL_GLOVE_DIR.mkdir(parents=True, exist_ok=True)

    if LOCAL_GLOVE_FILE.exists():
        print(f"Using local GloVe file: {LOCAL_GLOVE_FILE}")
        return LOCAL_GLOVE_FILE

    print("Local GloVe file not found. Downloading via kagglehub...")
    path = kagglehub.dataset_download("watts2/glove6b50dtxt")
    downloaded_file = find_glove_file(path)
    shutil.copy2(downloaded_file, LOCAL_GLOVE_FILE)
    print(f"GloVe file saved locally at: {LOCAL_GLOVE_FILE}")
    return LOCAL_GLOVE_FILE


def load_glove_embeddings(file_path: str | Path) -> Dict[str, torch.Tensor]:
    embeddings: Dict[str, torch.Tensor] = {}

    with open(file_path, "r", encoding="utf8") as handle:
        for line in handle:
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float32)
            embeddings[word] = vector

    return embeddings


def sentence_to_tensor(
    text: str,
    glove: Dict[str, torch.Tensor],
    embed_dim: int = EMBED_DIM,
) -> torch.Tensor:
    vectors = []

    for token in tokenize(text):
        if token in glove:
            vectors.append(glove[token])
        else:
            vectors.append(torch.zeros(embed_dim))

    return torch.stack(vectors)


def build_training_data(glove: Dict[str, torch.Tensor]):
    items = []

    for text, label in DATASET:
        x = sentence_to_tensor(text, glove, EMBED_DIM)
        y = torch.tensor(float(label), dtype=torch.float32)
        items.append((text, x, y))

    return items
