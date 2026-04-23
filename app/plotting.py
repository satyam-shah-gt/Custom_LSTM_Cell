from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .config import DEVICE, EMBED_DIM
from .data import sentence_to_tensor, tokenize


def _save_plot(output_path: Path):
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_dynamics(history, plots_dir: Path):
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["loss"], label="Loss")
    plt.plot(epochs, history["accuracy"], label="Accuracy")
    plt.title("Training loss and accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    _save_plot(plots_dir / "training_loss_accuracy.png")

    plt.figure()
    for key, label in (
        ("Wf", "Forget gate"),
        ("Wi", "Input gate"),
        ("Wg", "Candidate gate"),
        ("Wo", "Output gate"),
    ):
        plt.plot(epochs, history["grad_stats"][key], label=label)
    plt.title("Gradient magnitude per gate")
    plt.xlabel("Epoch")
    plt.ylabel("Average gradient magnitude")
    plt.legend()
    _save_plot(plots_dir / "training_gradients_per_gate.png")

    plt.figure()
    for key, label in (
        ("Wf", "Forget gate"),
        ("Wi", "Input gate"),
        ("Wg", "Candidate gate"),
        ("Wo", "Output gate"),
    ):
        plt.plot(epochs, history["weight_stats"][key], label=label)
    plt.title("Weight magnitude growth per gate")
    plt.xlabel("Epoch")
    plt.ylabel("Average |weight|")
    plt.legend()
    _save_plot(plots_dir / "training_weight_magnitude.png")

    plt.figure()
    for key, label in (
        ("Wf", "Forget gate"),
        ("Wi", "Input gate"),
        ("Wg", "Candidate gate"),
        ("Wo", "Output gate"),
    ):
        plt.plot(epochs, history["update_stats"][key], label=label)
    plt.title("Average weight change per epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Average |delta weight|")
    plt.legend()
    _save_plot(plots_dir / "training_weight_updates.png")

    plt.figure()
    plt.plot(epochs, history["gate_activation_stats"]["forget"], label="Forget gate")
    plt.plot(epochs, history["gate_activation_stats"]["input"], label="Input gate")
    plt.plot(epochs, history["gate_activation_stats"]["output"], label="Output gate")
    plt.title("Gate activation behaviour during training")
    plt.xlabel("Epoch")
    plt.ylabel("Average activation")
    plt.legend()
    _save_plot(plots_dir / "training_gate_activations.png")


def plot_gate_heatmap(
    model: nn.Module,
    sentence: str,
    glove: Dict[str, torch.Tensor],
    output_path: Path,
):
    model.eval()
    x = sentence_to_tensor(sentence, glove, EMBED_DIM).to(DEVICE)

    with torch.no_grad():
        _, gate_history = model(x, return_gates=True)

    matrix = np.array(
        [
            [g["forget"].mean().item() for g in gate_history],
            [g["input"].mean().item() for g in gate_history],
            [g["output"].mean().item() for g in gate_history],
        ]
    )

    plt.figure()
    plt.imshow(matrix)
    plt.yticks([0, 1, 2], ["Forget", "Input", "Output"])
    plt.xticks(range(len(tokenize(sentence))), tokenize(sentence), rotation=45)
    plt.title("Gate activation per word")
    plt.colorbar()
    _save_plot(output_path)


def plot_cell_state_dynamics(
    model: nn.Module,
    sentence: str,
    glove: Dict[str, torch.Tensor],
    output_path: Path,
):
    model.eval()
    x = sentence_to_tensor(sentence, glove, EMBED_DIM).to(DEVICE)

    with torch.no_grad():
        _, gate_history = model(x, return_gates=True)

    cell_means = [g["cell"].mean().item() for g in gate_history]

    plt.figure()
    plt.plot(cell_means)
    plt.xticks(range(len(tokenize(sentence))), tokenize(sentence), rotation=45)
    plt.title("Cell state evolution (memory change)")
    plt.xlabel("Word position")
    plt.ylabel("Average cell value")
    _save_plot(output_path)


def plot_hidden_state_dynamics(
    model: nn.Module,
    sentence: str,
    glove: Dict[str, torch.Tensor],
    output_path: Path,
):
    model.eval()
    x = sentence_to_tensor(sentence, glove, EMBED_DIM).to(DEVICE)

    with torch.no_grad():
        _, gate_history = model(x, return_gates=True)

    hidden_means = [g["hidden"].mean().item() for g in gate_history]

    plt.figure()
    plt.plot(hidden_means)
    plt.xticks(range(len(tokenize(sentence))), tokenize(sentence), rotation=45)
    plt.title("Hidden state evolution")
    plt.xlabel("Word position")
    plt.ylabel("Average hidden value")
    _save_plot(output_path)
