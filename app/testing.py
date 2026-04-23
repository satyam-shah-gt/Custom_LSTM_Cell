from typing import Dict

import torch
import torch.nn as nn

from .config import DEVICE, EMBED_DIM
from .data import sentence_to_tensor, tokenize


def inspect_sentence(model: nn.Module, sentence: str, glove: Dict[str, torch.Tensor]):
    model.eval()

    x = sentence_to_tensor(sentence, glove, EMBED_DIM).to(DEVICE)

    with torch.no_grad():
        logit, gate_history = model(x, return_gates=True)
        prob = torch.sigmoid(logit).item()

    predicted = "POSITIVE" if prob >= 0.5 else "NEGATIVE"

    print("\n" + "=" * 70)
    print(f"Sentence: {sentence}")
    print(f"Prediction: {predicted} | Probability: {prob:.4f}")
    print("=" * 70)

    for index, token in enumerate(tokenize(sentence)):
        gates = gate_history[index]

        print(f"\nTime step {index + 1} | Word: {token}")
        print("-" * 50)
        print(f"Forget gate mean   : {gates['forget'].mean().item():.4f}")
        print(f"Input gate mean    : {gates['input'].mean().item():.4f}")
        print(f"Output gate mean   : {gates['output'].mean().item():.4f}")
        print(f"Cell state mean    : {gates['cell'].mean().item():.4f}")
        print(f"Hidden state mean  : {gates['hidden'].mean().item():.4f}")

        print("Forget gate vector :", gates["forget"][:8].numpy())
        print("Input gate vector  :", gates["input"][:8].numpy())
        print("Output gate vector :", gates["output"][:8].numpy())
        print("Cell state vector  :", gates["cell"][:8].numpy())


def analyze_token_influence(model: nn.Module, sentence: str, glove: Dict[str, torch.Tensor]):
    model.eval()

    x = sentence_to_tensor(sentence, glove, EMBED_DIM).to(DEVICE)

    with torch.no_grad():
        _, gate_history = model(x, return_gates=True)

    print("\nTOKEN LEVEL GATE BEHAVIOUR")
    print("=" * 60)

    for index, token in enumerate(tokenize(sentence)):
        gates = gate_history[index]

        forget_strength = gates["forget"].mean().item()
        input_strength = gates["input"].mean().item()
        output_strength = gates["output"].mean().item()
        candidate_strength = gates["candidate"].abs().mean().item()

        print(f"\nWORD: {token}")
        print(f"forget gate influence      : {forget_strength:.3f}")
        print(f"input gate influence       : {input_strength:.3f}")
        print(f"candidate signal strength  : {candidate_strength:.3f}")
        print(f"output gate influence      : {output_strength:.3f}")
        print("interpretation:")

        if input_strength > 0.6:
            print(" strong memory update triggered")
        if forget_strength < 0.4:
            print(" previous memory partly discarded")
        if output_strength > 0.6:
            print(" strongly influencing prediction")
        if candidate_strength > 0.6:
            print(" strong semantic signal detected")
