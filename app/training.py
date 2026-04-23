import torch
import torch.nn as nn
import torch.optim as optim

from .config import DEVICE, EPOCHS, LEARNING_RATE, PRINT_EVERY


GATE_KEYS = ("Wf", "Wi", "Wg", "Wo")


def _empty_history():
    return {
        "loss": [],
        "accuracy": [],
        "grad_stats": {key: [] for key in GATE_KEYS},
        "weight_stats": {key: [] for key in GATE_KEYS},
        "update_stats": {key: [] for key in GATE_KEYS},
        "gate_activation_stats": {
            "forget": [],
            "input": [],
            "output": [],
        },
    }


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _safe_grad_mean(param):
    if param.grad is None:
        return 0.0
    return param.grad.detach().abs().mean().item()


def _safe_weight_mean(param):
    return param.detach().abs().mean().item()


def _collect_weight_snapshot(model):
    cell = model.lstm_cell
    return {key: getattr(cell, key).detach().clone() for key in GATE_KEYS}


def _update_magnitude(before_snapshot, model):
    cell = model.lstm_cell
    return {
        key: (getattr(cell, key).detach() - before_snapshot[key]).abs().mean().item()
        for key in GATE_KEYS
    }


def print_single_step_walkthrough(model: nn.Module, train_item):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    text, x, y = train_item
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    print("\nSingle training-step walkthrough")
    print("=" * 60)
    print(f"Sentence: {text}")
    print(f"Target label: {int(y.item())}")

    model.train()
    optimizer.zero_grad()

    before_update = _collect_weight_snapshot(model)
    logit, gate_history = model(x, return_gates=True)
    loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
    loss.backward()

    print(f"Loss before update: {loss.item():.4f}")
    print("\nPer-token gate activity before the optimizer step")
    print("-" * 60)

    tokens = text.split()
    for index, token in enumerate(tokens):
        gates = gate_history[index]
        print(
            f"{index + 1}. {token:<12}"
            f" forget={gates['forget'].mean().item():.3f}"
            f" input={gates['input'].mean().item():.3f}"
            f" candidate={gates['candidate'].mean().item():.3f}"
            f" output={gates['output'].mean().item():.3f}"
        )

    print("\nGradient magnitude for this training example")
    print("-" * 60)
    for key in GATE_KEYS:
        print(f"{key}: {_safe_grad_mean(getattr(model.lstm_cell, key)):.6f}")

    optimizer.step()
    update_stats = _update_magnitude(before_update, model)

    print("\nAverage weight change caused by this optimizer step")
    print("-" * 60)
    for key in GATE_KEYS:
        print(f"{key}: {update_stats[key]:.6f}")


def train_model(model: nn.Module, train_data):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = _empty_history()

    model.train()
    print("\nTracking learning dynamics per gate\n")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0

        epoch_grad_stats = {key: [] for key in GATE_KEYS}
        epoch_weight_stats = {key: [] for key in GATE_KEYS}
        epoch_update_stats = {key: [] for key in GATE_KEYS}
        epoch_gate_activation_stats = {
            "forget": [],
            "input": [],
            "output": [],
        }

        for _, x, y in train_data:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            before_update = _collect_weight_snapshot(model)

            logit, gates = model(x, return_gates=True)
            loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
            loss.backward()

            for key in GATE_KEYS:
                param = getattr(model.lstm_cell, key)
                epoch_grad_stats[key].append(_safe_grad_mean(param))
                epoch_weight_stats[key].append(_safe_weight_mean(param))

            for step_gate in gates:
                epoch_gate_activation_stats["forget"].append(
                    step_gate["forget"].abs().mean().item()
                )
                epoch_gate_activation_stats["input"].append(
                    step_gate["input"].abs().mean().item()
                )
                epoch_gate_activation_stats["output"].append(
                    step_gate["output"].abs().mean().item()
                )

            optimizer.step()

            step_update_stats = _update_magnitude(before_update, model)
            for key in GATE_KEYS:
                epoch_update_stats[key].append(step_update_stats[key])

            total_loss += loss.item()

            pred = torch.sigmoid(logit).item()
            predicted_label = 1 if pred >= 0.5 else 0
            if predicted_label == int(y.item()):
                correct += 1

        avg_loss = total_loss / len(train_data)
        acc = correct / len(train_data)

        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        for key in GATE_KEYS:
            history["grad_stats"][key].append(_mean(epoch_grad_stats[key]))
            history["weight_stats"][key].append(_mean(epoch_weight_stats[key]))
            history["update_stats"][key].append(_mean(epoch_update_stats[key]))

        for key in history["gate_activation_stats"]:
            history["gate_activation_stats"][key].append(
                _mean(epoch_gate_activation_stats[key])
            )

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print("\n==============================")
            print(f"Epoch {epoch}")
            print("==============================")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Accuracy: {acc:.2%}")

            print("\nGradient magnitude per gate")
            print("----------------------------")
            for key in GATE_KEYS:
                print(f"{key}: {history['grad_stats'][key][-1]}")

            print("\nAverage gate activation")
            print("----------------------------")
            print(f"Forget activation: {history['gate_activation_stats']['forget'][-1]}")
            print(f"Input activation: {history['gate_activation_stats']['input'][-1]}")
            print(f"Output activation: {history['gate_activation_stats']['output'][-1]}")

            print("\nWeight magnitude")
            print("----------------------------")
            for key in GATE_KEYS:
                print(f"{key}: {history['weight_stats'][key][-1]}")

            print("\nAverage weight update per training example")
            print("----------------------------")
            for key in GATE_KEYS:
                print(f"{key}: {history['update_stats'][key][-1]}")

    return history
