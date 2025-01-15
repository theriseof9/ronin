
import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
# Ensure that the ODELSTM classes are defined or imported
# In this example, they are defined within the script

from utils import load_config, MSEAverageMeter
from data_glob_speed import GlobSpeedSequence, SequenceToSequenceDataset
from transformations import ComposeTransform, RandomHoriRotateSeq
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error

torch.multiprocessing.set_sharing_strategy('file_system')

_nano_to_sec = 1e09
_input_channel, _output_channel = 6, 2  # Adjust based on your data
device = 'cpu'


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

    def get_receptive_field(self):
        # ODELSTM processes the entire sequence
        return -1  # Indicates the entire sequence is used


class GlobalPosLoss(torch.nn.Module):
    def __init__(self, mode='full', history=None):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalPosLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        assert mode in ['full', 'part']
        self.mode = mode
        if self.mode == 'part':
            assert history is not None
            self.history = history
        elif self.mode == 'full':
            self.history = 1

    def forward(self, pred, targ):
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        if self.mode == 'part':
            gt_pos = gt_pos[:, self.history:, :] - gt_pos[:, :-self.history, :]
            pred_pos = pred_pos[:, self.history:, :] - pred_pos[:, :-self.history, :]
        loss = self.mse_loss(pred_pos, gt_pos)
        return torch.mean(loss)


def write_config(args, **kwargs):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "odelstm_training"
            if kwargs:
                values['kwargs'] = kwargs
            json.dump(values, f, sort_keys=True)


def get_dataset(root_dir, data_list, args, **kwargs):
    input_format, output_format = [0, 3, 6], [0, _output_channel]
    mode = kwargs.get('mode', 'train')
    random_shift, shuffle, transforms, grv_only = 0, False, [], False
    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms.append(RandomHoriRotateSeq(input_format, output_format))
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True

    transforms = ComposeTransform(transforms)

    if args.dataset == 'ronin':
        seq_type = GlobSpeedSequence
    elif args.dataset == 'ridi':
        from data_ridi import RIDIGlobSpeedSequence
        seq_type = RIDIGlobSpeedSequence

    dataset = SequenceToSequenceDataset(
        seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
        random_shift=random_shift, transform=transforms, shuffle=shuffle, grv_only=grv_only, **kwargs
    )
    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def get_model(args, **kwargs):
    hidden_size = args.hidden_size  # Hidden size for ODELSTM
    solver_type = args.solver_type  # Solver type for ODELSTM
    print("Initializing ODELSTM Network")
    network = ODELSTMSeqNetwork(
        input_size=_input_channel,
        out_size=_output_channel,
        batch_size=args.batch_size,
        device=device,
        hidden_size=hidden_size,
        solver_type=solver_type
    ).to(device)
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. Trainable parameters: {}'.format(pytorch_total_params))
    return network


def get_loss_function(history, args, **kwargs):
    # For ODELSTM, we can use the full sequence for loss computation
    criterion = GlobalPosLoss(mode='full')
    return criterion


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args, **kwargs):
    # Loading data
    start_t = time.time()

    train_dataset = get_dataset_from_list(args.data_dir, args.train_list, args, mode='train', **kwargs)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=True, drop_last=True)

    end_t = time.time()
    print('Training set loaded. Time usage: {:.3f}s'.format(end_t - start_t))

    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.data_dir, args.val_list, args, mode='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        print('Validation set loaded')

    global device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)

    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    # Initialize model
    network = get_model(args, **kwargs).to(device)

    criterion = get_loss_function(None, args, **kwargs)  # Using full sequence loss
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.75, verbose=True, eps=1e-12)

    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', False)
    log_file = None

    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        with open(osp.join(str(Path(args.continue_from).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)
        if device.type == 'cpu':
            checkpoints = torch.load(args.continue_from, map_location=lambda storage, location: storage)
        else:
            checkpoints = torch.load(args.continue_from, map_location={model_data['device']: args.device})
        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'))
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
        if kwargs.get('force_lr', False):
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))

    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()

            train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_loss = 0
            start_t = time.time()
            for bid, batch in tqdm(enumerate(train_loader), total=train_mini_batches):
                # Assume batch returns feat, targ, timespans, _, _
                feat, targ, _, _ = batch
                timespans = torch.ones(feat.size(0), feat.size(1), device=device)
                feat, targ, timespans = feat.to(device), targ.to(device), timespans.to(device)

                optimizer.zero_grad()
                predicted = network(feat, timespans)  # Pass timespans to network

                train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                loss = criterion(predicted, targ)
                train_loss += loss.cpu().detach().numpy()
                loss.backward()
                optimizer.step()

                step += 1

            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()

            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}'.format(
                    epoch, end_t - start_t, train_errs[epoch],
                    train_vel.get_channel_avg(), train_vel.get_total_avg()))
                log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch],
                                         *train_vel.get_channel_avg())

            saved_model = False
            if val_loader:
                network.eval()
                val_vel = MSEAverageMeter(3, [2], _output_channel)
                val_loss = 0
                with torch.no_grad():
                    for bid, batch in enumerate(val_loader):
                        feat, targ, timespans, _, _ = batch
                        feat, targ, timespans = feat.to(device), targ.to(device), timespans.to(device)
                        pred = network(feat, timespans)
                        val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
                        val_loss += criterion(pred, targ).cpu().detach().numpy()

                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss, *val_vel.get_channel_avg())
                if not quiet_mode:
                    print('Validation loss: {} vel_loss: {}/{:.6f}'.format(
                        val_loss, val_vel.get_channel_avg(), val_vel.get_total_avg()))

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({
                            'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()
                        }, model_path)
                        print('Best Validation Model saved to ', model_path)

            if use_scheduler:
                scheduler.step(val_loss)

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({
                    'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'loss': train_errs[epoch],
                    'optimizer_state_dict': optimizer.state_dict()
                }, model_path)
                print('Model saved to ', model_path)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)

            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break

    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({
            'model_state_dict': network.state_dict(),
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict()
        }, model_path)


def recon_traj_with_preds_global(dataset, preds, ind=None, seq_id=0, type='preds', **kwargs):
    ind = ind if ind is not None else np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=int)

    if type == 'gt':
        pos = dataset.gt_pos[seq_id][:, :2]
    else:
        ts = dataset.ts[seq_id]
        # Compute the global velocity from local velocity.
        dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
        pos = preds * dts
        pos[0, :] = dataset.gt_pos[seq_id][0, :2]
        pos = np.cumsum(pos, axis=0)
    veloc = preds
    ori = dataset.orientations[seq_id]

    return pos, veloc, ori


# python ronin/source/train_odelstm.py test --data_dir data/unseen_subjects_test_set/ --test_list ronin/lists/list_train_amended.txt --model_path lstmode2/checkpoints/icheckpoint_22.pt --out_dir lstmode2test --device cuda:0
def test(args, **kwargs):
    global device, _output_channel

    import matplotlib.pyplot as plt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        root_dir = osp.split(args.test_path)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        root_dir = args.data_dir if args.data_dir else osp.split(args.test_list)[0]
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')

    # Load the first sequence to update the input and output size
    _ = get_dataset(root_dir, [test_data_list[0]], args, mode='test')

    if args.out_dir and not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)

    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage)
    else:
        checkpoint = torch.load(args.model_path, map_location={model_data['device']: args.device})

    network = get_model(args, **kwargs)
    network.load_state_dict(checkpoint.get('model_state_dict'))
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))

    log_file = None
    if args.test_list and args.out_dir:
        log_file = osp.join(args.out_dir, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.model_path + '\n')
            f.write('Seq traj_len velocity ate rte\n')

    losses_vel = MSEAverageMeter(2, [1], _output_channel)
    ate_all, rte_all = [], []
    pred_per_min = 200 * 60

    seq_dataset = get_dataset(root_dir, test_data_list, args, mode='test', **kwargs)

    for idx, data in enumerate(test_data_list):
        assert data == osp.split(seq_dataset.data_path[idx])[1]

        feat, vel = seq_dataset.get_test_seq(idx)  # Adjusted to get timespans
        feat = torch.Tensor(feat).to(device)
        timespans = torch.ones(feat.size(0), feat.size(1), device=device)
        with torch.no_grad():
            preds = network(feat, timespans)
        preds = np.squeeze(preds.cpu().detach().numpy())
        preds = preds[-vel.shape[0]:, :_output_channel]
        ind = np.arange(vel.shape[0])
        vel_losses = np.mean((vel - preds) ** 2, axis=0)
        losses_vel.add(vel, preds)

        print('Reconstructing trajectory')
        pos_pred = recon_traj_with_preds_global(seq_dataset, preds, timespans.cpu().numpy(), ind=ind, type='pred', seq_id=idx)
        pos_gt = recon_traj_with_preds_global(seq_dataset, vel, timespans.cpu().numpy(), ind=ind, type='gt', seq_id=idx)

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, '{}_{}.npy'.format(data, 'odelstm')), np.concatenate([pos_pred, pos_gt], axis=1))

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
        else:
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)

        ate_all.append(ate)
        rte_all.append(rte)

        print('Sequence {}, Velocity loss {} / {}, ATE: {}, RTE:{}'.format(data, vel_losses, np.mean(vel_losses), ate, rte))
        log_line = format_string(data, np.mean(vel_losses), ate, rte)

        if not args.fast_test:
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']

            plt.figure('{}'.format(data), figsize=(16, 9))
            plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
            plt.plot(pos_pred[:, 0], pos_pred[:, 1])
            plt.plot(pos_gt[:, 0], pos_gt[:, 1])
            plt.title(data)
            plt.axis('equal')
            plt.legend(['Predicted', 'Ground truth'])

            plt.subplot2grid((kp, 2), (kp - 1, 0))
            plt.plot(pos_cum_error)
            plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])

            for i in range(kp):
                plt.subplot2grid((kp, 2), (i, 1))
                plt.plot(ind, preds[:, i])
                plt.plot(ind, vel[:, i])
                plt.legend(['Predicted', 'Ground truth'])
                plt.title('{}, error: {:.6f}'.format(targ_names[i], vel_losses[i]))
            plt.tight_layout()

            if args.show_plot:
                plt.show()
            if args.out_dir is not None and osp.isdir(args.out_dir):
                plt.savefig(osp.join(args.out_dir, '{}_{}.png'.format(data, 'odelstm')))

            plt.close('all')

        if log_file is not None:
            with open(log_file, 'a') as f:
                log_line += '\n'
                f.write(log_line)

    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)
    measure = format_string('ATE', 'RTE', sep='\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), sep='\t')
    print(measure, '\n', values)
    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure + '\n')
            f.write(values)


if __name__ == '__main__':
    """
    Run file with individual arguments or/and config file.
    If an argument appears in both config file and args, args is given precedence.
    """
    default_config_file = osp.abspath(osp.join(osp.abspath(__file__), '../../config/temporal_model_defaults.json'))

    import argparse

    parser = argparse.ArgumentParser(
        description="Run ODELSTM model in train/test mode. Optional configurations can be specified as --key [value..] pairs",
        add_help=True)
    parser.add_argument('--config', type=str, help='Configuration file [Default: {}]'.format(default_config_file),
                        default=default_config_file)

    # Common arguments
    parser.add_argument('--data_dir', type=str, help='Directory for data files if different from list path.')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, help='Cuda device (e.g., cuda:0) or cpu')
    parser.add_argument('--dataset', type=str, choices=['ronin', 'ridi'])

    # ODELSTM-specific arguments
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--solver_type', type=str, choices=['fixed_euler', 'fixed_heun', 'fixed_rk4'], default='fixed_rk4')

    mode = parser.add_subparsers(title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True

    # Train mode arguments
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str)
    train_cmd.add_argument('--val_list', type=str)
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int)
    train_cmd.add_argument('--save_interval', type=int)
    train_cmd.add_argument('--lr', '--learning_rate', type=float)


    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_path', type=str, default=None)
    test_cmd.add_argument('--test_list', type=str, default=None)
    test_cmd.add_argument('--model_path', type=str, default=None)
    test_cmd.add_argument('--fast_test', action='store_true')
    test_cmd.add_argument('--show_plot', action='store_true')

    args, unknown_args = parser.parse_known_args()

    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)
    print(args, kwargs)

    if args.mode == 'train':
        train(args, **kwargs)
    elif args.mode == 'test':
        test(args, **kwargs)
    else:
        raise NotImplementedError("Only train mode is implemented in this script for ODELSTM.")

    # python ronin/source/train_odelstm.py train --data_dir data/alldata --train_list ronin/lists/list_train_amended.txt --val_list list_val.txt --epochs 100 --batch_size 1024 --lr 0.001 --window_size 200 --step_size 10 --hidden_size 128 --solver_type fixed_rk4 --out_dir lstmode2