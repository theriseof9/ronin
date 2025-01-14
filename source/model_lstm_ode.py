import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, ode_hidden_size=64, solver='rk4'):
        super(ODELSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

        # Define the ODE function for the hidden state h
        self.ode_func = nn.Sequential(
            nn.Linear(hidden_size, ode_hidden_size),
            nn.Tanh(),
            nn.Linear(ode_hidden_size, hidden_size)
        )

        self.solver = solver

    def forward(self, input, hx, timespans):
        h, c = self.lstm_cell(input, hx)

        # Use torchdiffeq's odeint
        # h: (batch_size, hidden_size)
        batch_size = h.size(0)
        h_list = []
        for i in range(batch_size):
            h_i = h[i].unsqueeze(0)  # Shape: (1, hidden_size)
            timespan_i = torch.tensor([0, timespans[i]], device=h.device)
            h_i = odeint(self.ode_func, h_i, timespan_i, method=self.solver)
            h_i = h_i[-1]  # Take the final time point
            h_list.append(h_i)
        h = torch.cat(h_list, dim=0)
        return h, c

class ODELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, ode_hidden_size=64, solver='dopri5'):
        super(ODELSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(ODELSTMCell(layer_input_size, hidden_size, ode_hidden_size, solver))

    def forward(self, inputs, timespans, hx=None):
        """
        inputs: (batch_size, seq_len, input_size)
        timespans: (batch_size, seq_len) - time intervals for each time step in the sequence
        hx: optional initial hidden and cell states
        """
        batch_size, seq_len, _ = inputs.size()

        if hx is None:
            h = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
            c = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        else:
            h, c = hx

        outputs = []
        for t in range(seq_len):
            input_t = inputs[:, t, :]  # (batch_size, input_size)
            timespan_t = timespans[:, t]  # (batch_size,)

            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](input_t, (h[layer], c[layer]), timespan_t)
                input_t = h[layer]  # Output of the current layer is input to the next layer

            outputs.append(h[-1].unsqueeze(1))  # Collect output from the last layer

        outputs = torch.cat(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        return outputs, (h, c)


class ODELSTMSeqNetwork(nn.Module):
    def __init__(self, input_size, out_size, lstm_size=100, lstm_layers=3, ode_hidden_size=64, solver='dopri5'):
        super(ODELSTMSeqNetwork, self).__init__()
        self.odelstm = ODELSTM(input_size, lstm_size, lstm_layers, ode_hidden_size, solver)
        self.linear1 = nn.Linear(lstm_size, out_size * 5)
        self.linear2 = nn.Linear(out_size * 5, out_size)

    def forward(self, inputs, timespans):
        """
        inputs: (batch_size, seq_len, input_size)
        timespans: (batch_size, seq_len) - time intervals for each time step
        """
        outputs, _ = self.odelstm(inputs, timespans)
        outputs = self.linear1(outputs)
        outputs = self.linear2(outputs)
        return outputs


class BilinearODELSTMSeqNetwork(nn.Module):
    def __init__(self, input_size, out_size, lstm_size=100, lstm_layers=3, ode_hidden_size=64, solver='dopri5'):
        super(BilinearODELSTMSeqNetwork, self).__init__()
        self.bilinear = nn.Bilinear(input_size, input_size, input_size * 4)
        self.odelstm = ODELSTM(input_size * 5, lstm_size, lstm_layers, ode_hidden_size, solver)
        self.linear1 = nn.Linear(lstm_size + input_size * 5, out_size * 5)
        self.linear2 = nn.Linear(out_size * 5, out_size)

    def forward(self, inputs, timespans):
        """
        inputs: (batch_size, seq_len, input_size)
        timespans: (batch_size, seq_len)
        """
        # Apply bilinear layer
        bilinear_out = self.bilinear(inputs, inputs)
        # Concatenate original input and bilinear output
        input_mix = torch.cat([inputs, bilinear_out], dim=2)  # Shape: (batch_size, seq_len, input_size * 5)

        # Pass through ODELSTM
        outputs, _ = self.odelstm(input_mix, timespans)

        # Concatenate input_mix and outputs from ODELSTM
        output_mix = torch.cat([input_mix, outputs], dim=2)  # Shape: (batch_size, seq_len, lstm_size + input_size * 5)

        outputs = self.linear1(output_mix)
        outputs = self.linear2(outputs)
        return outputs