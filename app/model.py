import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        concat_dim = input_dim + hidden_dim

        self.Wf = nn.Parameter(torch.randn(hidden_dim, concat_dim) * 0.1)
        self.Wi = nn.Parameter(torch.randn(hidden_dim, concat_dim) * 0.1)
        self.Wg = nn.Parameter(torch.randn(hidden_dim, concat_dim) * 0.1)
        self.Wo = nn.Parameter(torch.randn(hidden_dim, concat_dim) * 0.1)

        self.bf = nn.Parameter(torch.zeros(hidden_dim))
        self.bi = nn.Parameter(torch.zeros(hidden_dim))
        self.bg = nn.Parameter(torch.zeros(hidden_dim))
        self.bo = nn.Parameter(torch.zeros(hidden_dim))

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ):
        combined = torch.cat([h_prev, x_t], dim=0)

        f_t = torch.sigmoid(self.Wf @ combined + self.bf)
        i_t = torch.sigmoid(self.Wi @ combined + self.bi)
        g_t = torch.tanh(self.Wg @ combined + self.bg)
        c_t = f_t * c_prev + i_t * g_t
        o_t = torch.sigmoid(self.Wo @ combined + self.bo)
        h_t = o_t * torch.tanh(c_t)

        gates = {
            "forget": f_t.detach().cpu(),
            "input": i_t.detach().cpu(),
            "candidate": g_t.detach().cpu(),
            "output": o_t.detach().cpu(),
            "cell": c_t.detach().cpu(),
            "hidden": h_t.detach().cpu(),
        }

        return h_t, c_t, gates


class SentimentLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_cell = CustomLSTMCell(input_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor, return_gates: bool = False):
        h = torch.zeros(self.hidden_dim, device=sequence.device)
        c = torch.zeros(self.hidden_dim, device=sequence.device)

        gate_history = []

        for step in range(sequence.size(0)):
            x_t = sequence[step]
            h, c, gates = self.lstm_cell(x_t, h, c)
            if return_gates:
                gate_history.append(gates)

        logit = self.classifier(h).squeeze(0)

        if return_gates:
            return logit, gate_history

        return logit
