import torch
import torch.nn as nn
from torchdiffeq import odeint

def add_time(x, t):
    bs, c, l = x.shape
    t_expanded = t.expand(bs, 1, l)
    xt = torch.cat([x, t_expanded], dim=1)
    return xt

def conv3(in_planes, out_planes, kernel_size, stride=1, dilation=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, bias=False)

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, dilation=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = conv3(in_planes, out_planes, kernel_size, stride, dilation)
        self.bn1 = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_planes, out_planes, kernel_size)
        self.bn2 = nn.BatchNorm1d(out_planes)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Conv1dODEFunc(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(Conv1dODEFunc, self).__init__()
        self.conv1 = nn.Conv1d(dim + 1, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(dim + 1, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, t, x):
        xt = add_time(x, t)
        out = self.conv1(xt)
        out = self.bn1(out)
        out = self.relu(out)
        ht = add_time(out, t)
        dxdt = self.conv2(ht)
        dxdt = self.bn2(dxdt)
        return dxdt

class ODEBlock(nn.Module):
    def __init__(self, odefunc, integration_time=torch.tensor([0, 1]).float()):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = integration_time

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, method='rk4')
        return out[1]
class FCOutputModule(nn.Module):
    """
    Fully connected output module.
    """
    def __init__(self, in_planes, num_outputs, **kwargs):
        """
        Constructor for a fully connected output layer.

        Args:
          in_planes: number of planes (channels) of the layer immediately proceeding the output module.
          num_outputs: number of output predictions.
          fc_dim: dimension of the fully connected layer.
          dropout: the keep probability of the dropout layer
          trans_planes: (optional) number of planes of the transition convolutional layer.
        """
        super(FCOutputModule, self).__init__()
        fc_dim = kwargs.get('fc_dim', 1024)
        dropout = kwargs.get('dropout', 0.5)
        in_dim = kwargs.get('in_dim', 7)
        trans_planes = kwargs.get('trans_planes', None)
        if trans_planes is not None:
            self.transition = nn.Sequential(
                nn.Conv1d(in_planes, trans_planes, kernel_size=1, bias=False),
                nn.BatchNorm1d(trans_planes))
            in_planes = trans_planes
        else:
            self.transition = None

        self.fc = nn.Sequential(
            nn.Linear(in_planes * in_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_outputs))

    def get_dropout(self):
        return [m for m in self.fc if isinstance(m, torch.nn.Dropout)]

    def forward(self, x):
        if self.transition is not None:
            x = self.transition(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y

class GlobAvgOutputModule(nn.Module):
    """ Global average output module. """
    def __init__(self, in_planes, num_outputs):
        super(GlobAvgOutputModule, self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_planes, num_outputs)
    
    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ResNet1D(nn.Module):
    def __init__(self, num_inputs, num_outputs, block_type, group_sizes, base_plane=64, output_block=None, zero_init_residual=False, **kwargs):
        super(ResNet1D, self).__init__()
        self.base_plane = base_plane
        self.inplanes = self.base_plane
        # Input Module
        self.input_block = nn.Sequential(
            nn.Conv1d(num_inputs, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        # ODE Blocks
        self.group_sizes = group_sizes  # Store group sizes
        self.planes = [self.base_plane * (2 ** i) for i in range(len(self.group_sizes))]
        kernel_size = kwargs.get('kernel_size', 3)
        groups = []
        in_planes = self.inplanes
        for i, (planes, num_layers) in enumerate(zip(self.planes, self.group_sizes)):
            stride = 1 if i == 0 else 2
            layer = self._make_ode_layer(in_planes, planes, kernel_size, stride, num_layers)
            groups.append(layer)
            in_planes = planes
        self.residual_groups = nn.Sequential(*groups)
        self.inplanes = in_planes  # Update self.inplanes
        # Output Module
        if output_block is None:
            self.output_block = GlobAvgOutputModule(self.inplanes, num_outputs)
        else:
            self.output_block = output_block(self.inplanes, num_outputs, **kwargs)
        self._initialize(zero_init_residual)

    def _make_ode_layer(self, in_planes, out_planes, kernel_size, stride=1, num_layers=1):
        layers = []
        if stride != 1 or in_planes != out_planes:
            prelayer = nn.Sequential(
                nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True)
            )
        else:
            prelayer = nn.Identity()
        layers.append(prelayer)
        for _ in range(num_layers):
            odefunc = Conv1dODEFunc(out_planes, kernel_size)
            odeblock = ODEBlock(odefunc)
            layers.append(odeblock)
        return nn.Sequential(*layers)
    
    def _initialize(self, zero_init_residual):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # Zero-initialize the last BN in ODE functions
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Conv1dODEFunc):
                    nn.init.constant_(m.bn2.weight, 0)  # Zero-initialize bn2 weight

    def forward(self, x):
        x = self.input_block(x)
        x = self.residual_groups(x)
        x = self.output_block(x)
        return x