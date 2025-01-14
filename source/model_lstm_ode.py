import torch
import torch.nn as nn
from torchdiffeq import odeint


class ODEFunc(nn.Module):
    def __init__(self, hidden_size, ode_hidden_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, ode_hidden_size),
            nn.Tanh(),
            nn.Linear(ode_hidden_size, hidden_size)
        )

    def forward(self, t, y):
        return self.net(y)

class ODELSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, ode_hidden_size=64, solver='rk4'):
        super(ODELSTMCell, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.solver = solver

        # Use the custom ODE function
        self.ode_func = ODEFunc(hidden_size, ode_hidden_size)

    def forward(self, input, hx, timespans):
        h, c = self.lstm_cell(input, hx)  # h: (batch_size, hidden_size)

        # timespans: (batch_size,)
        # We need to ensure that timespans are the same for all samples
        # For batch processing, we'll use a common timespan

        timespan = timespans[0].item()  # Assuming timespans are the same
        timespan_tensor = torch.tensor([0, timespan], device=h.device)

        h = odeint(self.ode_func, h, timespan_tensor, method=self.solver)
        h = h[-1]  # Take the final time point
        return h, c

class ODELSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, ode_hidden_size=64, solver='rk4'):
        super(ODELSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.cells.append(ODELSTMCell(layer_input_size, hidden_size, ode_hidden_size, solver))

    def forward(self, inputs, timespans, hx=None):
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

            # Ensure timespans are the same
            timespan = timespan_t[0].item()
            timespan_tensor = torch.tensor([0, timespan], device=inputs.device)

            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](input_t, (h[layer], c[layer]), timespan_t)
                input_t = h[layer]
            outputs.append(h[-1].unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
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