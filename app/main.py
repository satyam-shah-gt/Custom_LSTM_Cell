from .config import DEVICE, EMBED_DIM, HIDDEN_DIM, TRACE_SENTENCE
from .data import build_training_data, download_glove, load_glove_embeddings
from .io_utils import append_summary, create_run_paths, log_block
from .model import SentimentLSTM
from .plotting import (
    plot_cell_state_dynamics,
    plot_gate_heatmap,
    plot_hidden_state_dynamics,
    plot_training_dynamics,
)
from .testing import analyze_token_influence, inspect_sentence
from .training import print_single_step_walkthrough, print_token_trace_report, train_model


def _slugify(sentence: str) -> str:
    return "_".join(sentence.lower().split())


def main():
    run_paths = create_run_paths()
    log_file = run_paths["log_file"]
    summary_file = run_paths["summary_file"]
    plots_dir = run_paths["plots_dir"]

    print(f"Using device: {DEVICE}")
    print(f"Run folder: {run_paths['run_dir']}")
    append_summary(summary_file, f"Device: {DEVICE}")
    append_summary(summary_file, f"Run folder: {run_paths['run_dir']}")
    append_summary(summary_file, f"Detailed log: {log_file.name}")

    print("Phase 1/5: locating GloVe")
    glove_path = download_glove()
    glove = load_glove_embeddings(glove_path)
    print(f"GloVe ready: {glove_path}")
    append_summary(summary_file, f"GloVe file: {glove_path}")
    append_summary(summary_file, f"Loaded vectors: {len(glove)}")

    print("Phase 2/5: preparing training data")
    train_data = build_training_data(glove)
    append_summary(summary_file, f"Training samples: {len(train_data)}")
    append_summary(summary_file, f"Tracked sentence: {TRACE_SENTENCE}")

    print("Phase 3/5: saving one training-step walkthrough")
    demo_model = SentimentLSTM(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    log_block(log_file, print_single_step_walkthrough, demo_model, train_data[0])

    print("Phase 4/5: training model and saving training graphs")
    model = SentimentLSTM(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    history = log_block(log_file, train_model, model, train_data, TRACE_SENTENCE)
    log_block(log_file, print_token_trace_report, history)
    plot_training_dynamics(history, plots_dir)
    append_summary(summary_file, "Saved training plots:")
    append_summary(summary_file, "- training_loss_accuracy.png")
    append_summary(summary_file, "- training_gradients_per_gate.png")
    append_summary(summary_file, "- training_weight_magnitude.png")
    append_summary(summary_file, "- training_weight_updates.png")
    append_summary(summary_file, "- training_bias_gradients.png")
    append_summary(summary_file, "- training_bias_magnitude.png")
    append_summary(summary_file, "- training_bias_updates.png")
    append_summary(summary_file, "- training_gate_activations.png")
    append_summary(summary_file, "- trace_prediction_over_epochs.png")
    append_summary(summary_file, "- trace_forget_heatmap.png")
    append_summary(summary_file, "- trace_input_heatmap.png")
    append_summary(summary_file, "- trace_candidate_heatmap.png")
    append_summary(summary_file, "- trace_output_heatmap.png")
    append_summary(summary_file, "- trace_cell_heatmap.png")
    append_summary(summary_file, "- trace_hidden_heatmap.png")

    test_sentences = [
        "i love this movie",
        "this film was terrible",
        "i love this movie but the ending was terrible",
        "the acting was bad but the story was great",
        "the film was boring but the climax was fantastic",
    ]

    print("Phase 5/5: testing sentences and saving sentence graphs")
    for sentence in test_sentences:
        slug = _slugify(sentence)
        log_block(log_file, inspect_sentence, model, sentence, glove)
        log_block(log_file, analyze_token_influence, model, sentence, glove)
        plot_gate_heatmap(
            model,
            sentence,
            glove,
            plots_dir / f"{slug}_gate_heatmap.png",
        )
        plot_cell_state_dynamics(
            model,
            sentence,
            glove,
            plots_dir / f"{slug}_cell_state.png",
        )
        plot_hidden_state_dynamics(
            model,
            sentence,
            glove,
            plots_dir / f"{slug}_hidden_state.png",
        )
        append_summary(summary_file, f"Sentence analyzed: {sentence}")

    print("Run complete.")
    print(f"Detailed log saved to: {log_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Plots saved to: {plots_dir}")
