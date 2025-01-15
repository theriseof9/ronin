import torch
from torch.autograd import Variable

# Import the ODELSTM and ODELSTMCell classes
class ODELSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, solver_type="fixed_rk4"):
        super(ODELSTMCell, self).__init__()
        self.solver_type = solver_type
        self.fixed_step_solver = solver_type.startswith("fixed_")
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)
        # 1 hidden layer NODE
        self.f_node = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
        )
        self.input_size = input_size
        self.hidden_size = hidden_size

        options = {
            "fixed_euler": self.euler,
            "fixed_heun": self.heun,
            "fixed_rk4": self.rk4,
        }
        if not solver_type in options.keys():
            raise ValueError("Unknown solver type '{:}'".format(solver_type))
        self.node = options[self.solver_type]

    def forward(self, input, hx, ts):
        new_h, new_c = self.lstm(input, hx)
        if self.fixed_step_solver:
            new_h = self.solve_fixed(new_h, ts)
        else:
            raise NotImplementedError("Variable step solvers are not implemented in this code.")
        return (new_h, new_c)

    def solve_fixed(self, x, ts):
        delta_t = ts.view(-1, 1)
        for i in range(3):  # 3 unfolds
            x = self.node(x, delta_t * (1.0 / 3))
        return x

    def euler(self, y, delta_t):
        dy = self.f_node(y)
        return y + delta_t * dy

    def heun(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + delta_t * k1)
        return y + delta_t * 0.5 * (k1 + k2)

    def rk4(self, y, delta_t):
        k1 = self.f_node(y)
        k2 = self.f_node(y + k1 * delta_t * 0.5)
        k3 = self.f_node(y + k2 * delta_t * 0.5)
        k4 = self.f_node(y + k3 * delta_t)
        return y + delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

class ODELSTM(torch.nn.Module):
    def __init__(
        self, in_features, hidden_size, out_feature, return_sequences=True, solver_type="fixed_rk4",
    ):
        super(ODELSTM, self).__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_feature = out_feature
        self.return_sequences = return_sequences

        self.rnn_cell = ODELSTMCell(in_features, hidden_size, solver_type=solver_type)
        self.fc = torch.nn.Linear(self.hidden_size, self.out_feature)

    def forward(self, x, timespans, mask=None):
        device = x.device
        batch_size = x.size(0)
        seq_len = x.size(1)
        hidden_state = (
            torch.zeros((batch_size, self.hidden_size), device=device),
            torch.zeros((batch_size, self.hidden_size), device=device),
        )
        outputs = []
        last_output = torch.zeros((batch_size, self.out_feature), device=device)
        for t in range(seq_len):
            inputs = x[:, t]
            ts = timespans[:, t].squeeze()
            hidden_state = self.rnn_cell(inputs, hidden_state, ts)
            current_output = self.fc(hidden_state[0])
            outputs.append(current_output)
            if mask is not None:
                cur_mask = mask[:, t].view(batch_size, 1)
                last_output = cur_mask * current_output + (1.0 - cur_mask) * last_output
            else:
                last_output = current_output

        if self.return_sequences:
            outputs = torch.stack(outputs, dim=1)  # Return entire sequence
        else:
            outputs = last_output  # Only last item
        return outputs

class ODELSTMSeqNetwork(torch.nn.Module):
    def __init__(self, input_size, out_size, batch_size, device, hidden_size=100, solver_type="fixed_rk4", dropout=0):
        """
        ODELSTM network adapted from LSTMSeqNetwork.

        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: Number of input features
        :param out_size: Number of output features
        :param batch_size: Batch size
        :param device: Torch device
        :param hidden_size: Number of hidden units in ODELSTM
        :param solver_type: Type of ODE solver to use in ODELSTMCell
        :param dropout: Dropout probability (not used here but kept for compatibility)
        """
        super(ODELSTMSeqNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = out_size
        self.batch_size = batch_size
        self.device = device

        # Initialize the ODELSTM layer
        self.odelstm = ODELSTM(
            in_features=self.input_size,
            hidden_size=self.hidden_size,
            out_feature=self.hidden_size,  # ODELSTM outputs hidden_size features
            return_sequences=True,
            solver_type=solver_type
        )

        # Linear layers to map from hidden size to output size
        self.linear1 = torch.nn.Linear(self.hidden_size, self.output_size * 5)
        self.linear2 = torch.nn.Linear(self.output_size * 5, self.output_size)

    def forward(self, input, timespans=None):
        """
        Forward pass of the network.

        :param input: Input tensor of shape [batch, frames, input_size]
        :param timespans: Tensor of time intervals between frames [batch, frames]. If None, assumes constant intervals.
        :return: Output tensor of shape [batch, frames, output_size]
        """
        if timespans is None:
            # Assume constant time intervals
            batch_size, seq_len, _ = input.size()
            timespans = torch.ones((batch_size, seq_len), device=self.device)

        # Pass the input through the ODELSTM layer
        output = self.odelstm(input, timespans)

        # Apply the linear layers
        output = self.linear1(output)
        output = self.linear2(output)

        return output