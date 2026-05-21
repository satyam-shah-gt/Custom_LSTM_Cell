import torch
import torch.nn as nn
import torch.optim as optim

from .config import DEVICE, EPOCHS, LEARNING_RATE, PRINT_EVERY
from .data import tokenize


WEIGHT_KEYS = ("Wf", "Wi", "Wg", "Wo")
BIAS_KEYS = ("bf", "bi", "bg", "bo")


def _empty_history():
    return {
        "loss": [],
        "accuracy": [],
        "grad_stats": {key: [] for key in WEIGHT_KEYS},
        "weight_stats": {key: [] for key in WEIGHT_KEYS},
        "update_stats": {key: [] for key in WEIGHT_KEYS},
        "bias_grad_stats": {key: [] for key in BIAS_KEYS},
        "bias_stats": {key: [] for key in BIAS_KEYS},
        "bias_update_stats": {key: [] for key in BIAS_KEYS},
        "gate_activation_stats": {
            "forget": [],
            "input": [],
            "output": [],
        },
        "token_trace": {
            "sentence": "",
            "tokens": [],
            "epochs": [],
            "probability": [],
            "loss": [],
            "forget": [],
            "input": [],
            "candidate": [],
            "output": [],
            "cell": [],
            "hidden": [],
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


def _collect_param_snapshot(model, keys):
    cell = model.lstm_cell
    return {key: getattr(cell, key).detach().clone() for key in keys}


def _update_magnitude(before_snapshot, model, keys):
    cell = model.lstm_cell
    return {
        key: (getattr(cell, key).detach() - before_snapshot[key]).abs().mean().item()
        for key in keys
    }


def _capture_token_trace(model: nn.Module, trace_item, criterion):
    text, x, y = trace_item
    x = x.to(DEVICE)
    y = y.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logit, gate_history = model(x, return_gates=True)
        prob = torch.sigmoid(logit).item()
        loss = criterion(logit.unsqueeze(0), y.unsqueeze(0)).item()

    trace = {
        "probability": prob,
        "loss": loss,
        "forget": [],
        "input": [],
        "candidate": [],
        "output": [],
        "cell": [],
        "hidden": [],
    }

    for gates in gate_history:
        trace["forget"].append(gates["forget"].mean().item())
        trace["input"].append(gates["input"].mean().item())
        trace["candidate"].append(gates["candidate"].mean().item())
        trace["output"].append(gates["output"].mean().item())
        trace["cell"].append(gates["cell"].mean().item())
        trace["hidden"].append(gates["hidden"].mean().item())

    return trace


def print_token_trace_report(history):
    trace = history["token_trace"]

    print("\nTracked sentence across epochs")
    print("=" * 60)
    print(f"Sentence: {trace['sentence']}")
    print(f"Tokens: {', '.join(trace['tokens'])}")

    for index, epoch in enumerate(trace["epochs"]):
        print(f"\nEpoch {epoch}")
        print(f"Probability: {trace['probability'][index]:.4f}")
        print(f"Trace loss: {trace['loss'][index]:.4f}")

        for token_index, token in enumerate(trace["tokens"]):
            print(
                f"{token_index + 1}. {token:<12}"
                f" forget={trace['forget'][index][token_index]:.3f}"
                f" input={trace['input'][index][token_index]:.3f}"
                f" candidate={trace['candidate'][index][token_index]:.3f}"
                f" output={trace['output'][index][token_index]:.3f}"
                f" cell={trace['cell'][index][token_index]:.3f}"
                f" hidden={trace['hidden'][index][token_index]:.3f}"
            )


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

    before_update = _collect_param_snapshot(model, WEIGHT_KEYS)
    before_bias_update = _collect_param_snapshot(model, BIAS_KEYS)
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
    for key in WEIGHT_KEYS:
        print(f"{key}: {_safe_grad_mean(getattr(model.lstm_cell, key)):.6f}")

    print("\nBias gradient magnitude for this training example")
    print("-" * 60)
    for key in BIAS_KEYS:
        print(f"{key}: {_safe_grad_mean(getattr(model.lstm_cell, key)):.6f}")

    optimizer.step()
    update_stats = _update_magnitude(before_update, model, WEIGHT_KEYS)
    bias_update_stats = _update_magnitude(before_bias_update, model, BIAS_KEYS)

    print("\nAverage weight change caused by this optimizer step")
    print("-" * 60)
    for key in WEIGHT_KEYS:
        print(f"{key}: {update_stats[key]:.6f}")

    print("\nAverage bias change caused by this optimizer step")
    print("-" * 60)
    for key in BIAS_KEYS:
        print(f"{key}: {bias_update_stats[key]:.6f}")


def train_model(model: nn.Module, train_data, trace_sentence: str | None = None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    history = _empty_history()
    trace_item = None

    if trace_sentence is not None:
        for item in train_data:
            if item[0] == trace_sentence:
                trace_item = item
                break
    if trace_item is None:
        trace_item = train_data[0]

    history["token_trace"]["sentence"] = trace_item[0]
    history["token_trace"]["tokens"] = tokenize(trace_item[0])

    model.train()
    print("\nTracking learning dynamics per gate\n")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        correct = 0

        epoch_grad_stats = {key: [] for key in WEIGHT_KEYS}
        epoch_weight_stats = {key: [] for key in WEIGHT_KEYS}
        epoch_update_stats = {key: [] for key in WEIGHT_KEYS}
        epoch_bias_grad_stats = {key: [] for key in BIAS_KEYS}
        epoch_bias_stats = {key: [] for key in BIAS_KEYS}
        epoch_bias_update_stats = {key: [] for key in BIAS_KEYS}
        epoch_gate_activation_stats = {
            "forget": [],
            "input": [],
            "output": [],
        }

        for _, x, y in train_data:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            before_update = _collect_param_snapshot(model, WEIGHT_KEYS)
            before_bias_update = _collect_param_snapshot(model, BIAS_KEYS)

            logit, gates = model(x, return_gates=True)
            loss = criterion(logit.unsqueeze(0), y.unsqueeze(0))
            loss.backward()

            for key in WEIGHT_KEYS:
                param = getattr(model.lstm_cell, key)
                epoch_grad_stats[key].append(_safe_grad_mean(param))
                epoch_weight_stats[key].append(_safe_weight_mean(param))

            for key in BIAS_KEYS:
                param = getattr(model.lstm_cell, key)
                epoch_bias_grad_stats[key].append(_safe_grad_mean(param))
                epoch_bias_stats[key].append(_safe_weight_mean(param))

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

            step_update_stats = _update_magnitude(before_update, model, WEIGHT_KEYS)
            step_bias_update_stats = _update_magnitude(
                before_bias_update, model, BIAS_KEYS
            )
            for key in WEIGHT_KEYS:
                epoch_update_stats[key].append(step_update_stats[key])
            for key in BIAS_KEYS:
                epoch_bias_update_stats[key].append(step_bias_update_stats[key])

            total_loss += loss.item()

            pred = torch.sigmoid(logit).item()
            predicted_label = 1 if pred >= 0.5 else 0
            if predicted_label == int(y.item()):
                correct += 1

        avg_loss = total_loss / len(train_data)
        acc = correct / len(train_data)

        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        for key in WEIGHT_KEYS:
            history["grad_stats"][key].append(_mean(epoch_grad_stats[key]))
            history["weight_stats"][key].append(_mean(epoch_weight_stats[key]))
            history["update_stats"][key].append(_mean(epoch_update_stats[key]))

        for key in BIAS_KEYS:
            history["bias_grad_stats"][key].append(_mean(epoch_bias_grad_stats[key]))
            history["bias_stats"][key].append(_mean(epoch_bias_stats[key]))
            history["bias_update_stats"][key].append(_mean(epoch_bias_update_stats[key]))

        for key in history["gate_activation_stats"]:
            history["gate_activation_stats"][key].append(
                _mean(epoch_gate_activation_stats[key])
            )

        trace = _capture_token_trace(model, trace_item, criterion)
        history["token_trace"]["epochs"].append(epoch)
        history["token_trace"]["probability"].append(trace["probability"])
        history["token_trace"]["loss"].append(trace["loss"])
        for key in ("forget", "input", "candidate", "output", "cell", "hidden"):
            history["token_trace"][key].append(trace[key])

        if epoch % PRINT_EVERY == 0 or epoch == 1:
            print("\n==============================")
            print(f"Epoch {epoch}")
            print("==============================")
            print(f"Loss: {avg_loss:.4f}")
            print(f"Accuracy: {acc:.2%}")

            print("\nGradient magnitude per gate")
            print("----------------------------")
            for key in WEIGHT_KEYS:
                print(f"{key}: {history['grad_stats'][key][-1]}")

            print("\nBias gradient magnitude per gate")
            print("----------------------------")
            for key in BIAS_KEYS:
                print(f"{key}: {history['bias_grad_stats'][key][-1]}")

            print("\nAverage gate activation")
            print("----------------------------")
            print(f"Forget activation: {history['gate_activation_stats']['forget'][-1]}")
            print(f"Input activation: {history['gate_activation_stats']['input'][-1]}")
            print(f"Output activation: {history['gate_activation_stats']['output'][-1]}")

            print("\nWeight magnitude")
            print("----------------------------")
            for key in WEIGHT_KEYS:
                print(f"{key}: {history['weight_stats'][key][-1]}")

            print("\nBias magnitude")
            print("----------------------------")
            for key in BIAS_KEYS:
                print(f"{key}: {history['bias_stats'][key][-1]}")

            print("\nAverage weight update per training example")
            print("----------------------------")
            for key in WEIGHT_KEYS:
                print(f"{key}: {history['update_stats'][key][-1]}")

            print("\nAverage bias update per training example")
            print("----------------------------")
            for key in BIAS_KEYS:
                print(f"{key}: {history['bias_update_stats'][key][-1]}")

            print("\nTracked sentence probability")
            print("----------------------------")
            print(f"{history['token_trace']['probability'][-1]:.4f}")

    return history
