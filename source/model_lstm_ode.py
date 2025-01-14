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


from model_temporal import LSTMSeqNetwork, BilinearLSTMSeqNetwork, TCNSeqNetwork
from utils import load_config, MSEAverageMeter
from data_glob_speed import GlobSpeedSequence, SequenceToSequenceDataset
from transformations import ComposeTransform, RandomHoriRotateSeq
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error

'''
Temporal models with loss functions in global coordinate frame
Configurations
    - Model types 
        TCN - type=tcn
        LSTM_simple - type=lstm, lstm_bilinear        
'''

torch.multiprocessing.set_sharing_strategy('file_system')
_nano_to_sec = 1e09
_input_channel, _output_channel = 6, 2
device = 'cpu'


def create_odelstm_config(args):
    from neuralhydrology.utils.config import Config

    # Parse comma-separated lists into appropriate data structures
    use_frequencies = args.use_frequencies.split(',')
    predict_last_n = {freq: int(n) for freq, n in zip(use_frequencies, args.predict_last_n.split(','))}
    seq_length = {freq: int(n) for freq, n in zip(use_frequencies, args.seq_length.split(','))}

    config_dict = {
        'hidden_size': args.hidden_size,
        'use_frequencies': use_frequencies,
        'predict_last_n': predict_last_n,
        'seq_length': seq_length,
        'ode_random_freq_lower_bound': args.ode_random_freq_lower_bound,
        'ode_num_unfolds': args.ode_num_unfolds,
        'ode_method': args.ode_method,
        # Other necessary parameters
        # ...
    }
    cfg = Config(config_dict)
    return cfg
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
            values['file'] = "pytorch_global_position"
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
    dataset = SequenceToSequenceDataset(seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
                                        random_shift=random_shift, transform=transforms, shuffle=shuffle,
                                        grv_only=grv_only, **kwargs)

    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def get_model(args, **kwargs):
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')

    if args.type == 'tcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, args.kernel_size,
                                layer_channels=args.channels, **config)
        print("TCN Network. Receptive field: {} ".format(network.get_receptive_field()))
    elif args.type == 'lstm_bi':
        print("Bilinear LSTM Network")
        network = BilinearLSTMSeqNetwork(_input_channel, _output_channel, args.batch_size, device,
                                         lstm_layers=args.layers, lstm_size=args.layer_size, **config).to(device)

    elif args.type == 'odelstm':
        print("ODELSTM Network")
        from neuralhydrology.modelzoo.odelstm import ODELSTM
        cfg = create_odelstm_config(args)
        network = ODELSTM(cfg).to(device)

    else:
        print("Simple LSTM Network")
        network = LSTMSeqNetwork(
            _input_channel, _output_channel, args.batch_size, device,
            lstm_layers=args.layers, lstm_size=args.layer_size, **config
        ).to(device)

    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return network


def get_loss_function(history, args, **kwargs):
    if args.type == 'tcn':
        config = {'mode': 'part',
                  'history': history}
    else:
        config = {'mode': 'full'}

    criterion = GlobalPosLoss(**config)
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


def prepare_odelstm_data(batch, frequencies, cfg, device):
    """
    Extract inputs and targets from the batch and organize them
    into the dictionary format expected by ODELSTM.

    Parameters:
    - batch: The input batch from the DataLoader.
    - frequencies: A list of frequencies used by the ODELSTM model, sorted from lowest to highest frequency.
    - cfg: Configuration object containing model and training settings.
    - device: The device (CPU or GPU) to move the tensors to.

    Returns:
    - data: A dictionary containing the inputs and targets per frequency.
    """

    # Unpacking the batch
    # Assuming batch is a tuple: (inputs, targets, static_attributes, other_data)
    # Adjust this based on your actual batch structure
    inputs, targets, static_attrs, _ = batch

    # Move inputs, targets, and static_attrs to the device
    inputs = inputs.to(device)        # Shape: (batch_size, seq_length_high, input_size)
    targets = targets.to(device)      # Shape: (batch_size, seq_length_high, output_size)
    static_attrs = static_attrs.to(device)  # Shape: (batch_size, num_static_features)

    # Initialize the data dictionary
    data = {}

    # The highest frequency is assumed to have the smallest interval (e.g., 'hourly')
    highest_freq = frequencies[-1]

    # We'll need the frequency factors to downsample or aggregate data
    frequency_factors = cfg._frequency_factors  # Assuming cfg has this attribute
    slice_timesteps = cfg._slice_timesteps     # Assuming cfg has this attribute

    # Prepare data for each frequency
    for i, freq in enumerate(frequencies):
        freq_suffix = f"_{freq}"
        factor = frequency_factors[freq]  # Factor relative to the lowest frequency (cfg._frequencies[0])

        # Determine the sequence length for the current frequency
        seq_length = cfg.seq_length[freq]

        # For the highest frequency, use the inputs and targets directly
        if freq == highest_freq:
            # Transpose to [seq_length, batch_size, input_size]
            x_d = inputs[:, -seq_length:, :].transpose(0, 1)  # Shape: [seq_length, batch_size, input_size]
            y = targets[:, -seq_length:, :].transpose(0, 1)   # Shape: [seq_length, batch_size, output_size]
        else:
            # For lower frequencies, downsample or aggregate the inputs and targets
            # Determine the indices to select from the high-frequency data
            step = factor
            x_d_high = inputs[:, -seq_length * step:, :]  # Shape: [batch_size, seq_length * step, input_size]
            y_high = targets[:, -seq_length * step:, :]   # Shape: [batch_size, seq_length * step, output_size]

            batch_size = x_d_high.size(0)
            # Reshape and aggregate over the time dimension
            x_d = x_d_high.reshape(batch_size, seq_length, step, -1).mean(dim=2)
            y = y_high.reshape(batch_size, seq_length, step, -1).mean(dim=2)

            # Transpose to [seq_length, batch_size, input_size]
            x_d = x_d.transpose(0, 1)
            y = y.transpose(0, 1)

        # Add dynamic inputs to the data dictionary
        data[f'x_d{freq_suffix}'] = x_d  # Shape: [seq_length, batch_size, input_size]

        # Add static inputs if any
        if static_attrs is not None:
            # Expand static attrs to match the sequence length
            x_s = static_attrs.unsqueeze(0).repeat(seq_length, 1, 1)
            data[f'x_s{freq_suffix}'] = x_s  # Shape: [seq_length, batch_size, num_static_features]

        # Add targets to the data dictionary
        data[f'y{freq_suffix}'] = y  # Shape: [seq_length, batch_size, output_size]

    # If using basin ID encoding or other one-hot encodings, include them
    if cfg.use_basin_id_encoding:
        data['x_one_hot'] = batch['x_one_hot'].to(device)  # Shape: [batch_size, num_basins]
        # Expand and repeat to match sequence length
        data['x_one_hot'] = data['x_one_hot'].unsqueeze(0).repeat(seq_length, 1, 1)

    return data
def compute_odelstm_loss(output, data, criterion):
    """
    Compute the loss over all frequencies.

    Parameters:
    - output: The model's output dictionary, with keys like 'y_hat_daily', 'y_hat_hourly', etc.
    - data: The data dictionary containing targets, with keys like 'y_daily', 'y_hourly', etc.
    - criterion: The loss function (e.g., nn.MSELoss()).

    Returns:
    - loss: The total loss summed over all frequencies.
    """
    loss = 0
    for freq in output.keys():
        freq_suffix = f"_{freq.split('_')[-1]}"
        y_hat = output[freq]  # Predicted outputs
        y_true = data[f'y{freq_suffix}']  # Ground truth targets
        # Reshape y_true if necessary to match y_hat
        # Compute loss for the current frequency
        loss += criterion(y_hat, y_true)
    return loss
def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    train_dataset = get_dataset_from_list(args.data_dir, args.train_list, args, mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                              drop_last=True)
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

    network = get_model(args, **kwargs).to(device)
    history = network.get_receptive_field() if args.type == 'tcn' else args.window_size // 2
    criterion = get_loss_function(history, args, **kwargs)

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
            for bid, batch in enumerate(train_loader):
                # Prepare the data dictionary expected by ODELSTM
                data = prepare_odelstm_data(batch)
                optimizer.zero_grad()
                output = network(data)
                train_vel.add(output.cpu().detach().numpy(), data['y_freq1'].cpu().detach().numpy())
                # The output is a dictionary with keys like 'y_hat_freq'
                # You need to aggregate losses over all frequencies
                loss = compute_odelstm_loss(output, data, criterion)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                step += 1
            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}'.format(
                    epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), train_vel.get_total_avg()))
            log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch],
                                     *train_vel.get_channel_avg())

            saved_model = False
            if val_loader:
                network.eval()
                val_vel = MSEAverageMeter(3, [2], _output_channel)
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feat, targ, _, _ = batch
                    feat, targ = feat.to(device), targ.to(device)
                    optimizer.zero_grad()
                    pred = network(feat)
                    val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
                    val_loss += criterion(pred, targ).cpu().detach().numpy()
                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss, *val_vel.get_channel_avg())
                if not quiet_mode:
                    print('Validation loss: {} vel_loss: {}/{:.6f}'.format(val_loss, val_vel.get_channel_avg(),
                                                                           val_vel.get_total_avg()))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(val_loss)

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:  # save even with validation
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
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
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)


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

        feat, vel = seq_dataset.get_test_seq(idx)
        feat = torch.Tensor(feat).to(device)
        preds = np.squeeze(network(feat).cpu().detach().numpy())[-vel.shape[0]:, :_output_channel]

        ind = np.arange(vel.shape[0])
        vel_losses = np.mean((vel - preds) ** 2, axis=0)
        losses_vel.add(vel, preds)

        print('Reconstructing trajectory')
        pos_pred, gv_pred, _ = recon_traj_with_preds_global(seq_dataset, preds, ind=ind, type='pred', seq_id=idx)
        pos_gt, gv_gt, _ = recon_traj_with_preds_global(seq_dataset, vel, ind=ind, type='gt', seq_id=idx)

        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, '{}_{}.npy'.format(data, args.type)),
                    np.concatenate([pos_pred, pos_gt], axis=1))

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
        else:
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
        ate_all.append(ate)
        rte_all.append(rte)

        print('Sequence {}, Velocity loss {} / {}, ATE: {}, RTE:{}'.format(data, vel_losses, np.mean(vel_losses), ate,
                                                                           rte))
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
                plt.savefig(osp.join(args.out_dir, '{}_{}.png'.format(data, args.type)))

        if log_file is not None:
            with open(log_file, 'a') as f:
                log_line += '\n'
                f.write(log_line)

        plt.close('all')

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
    Run file with individual arguments or/and config file. If argument appears in both config file and args, 
    args is given precedence.
    """
    default_config_file = osp.abspath(osp.join(osp.abspath(__file__), '../../config/temporal_model_defaults.json'))

    import argparse

    parser = argparse.ArgumentParser(description="Run seq2seq model in train/test mode [required]. Optional "
                                                 "configurations can be specified as --key [value..] pairs",
                                     add_help=True)
    parser.add_argument('--config', type=str, help='Configuration file [Default: {}]'.format(default_config_file),
                        default=default_config_file)
    # common

    parser.add_argument('--type', type=str, choices=['tcn', 'lstm', 'lstm_bi', 'odelstm'],
                        help='Model type')
    parser.add_argument('--data_dir', type=str, help='Directory for data files if different from list path.')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, help='Gaussian for smoothing features')
    parser.add_argument('--target_sigma', type=float, help='Gaussian for smoothing target')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, help='Cuda device (e.g:- cuda:0) or cpu')
    parser.add_argument('--dataset', type=str, choices=['ronin', 'ridi'])
    # tcn
    tcn_cmd = parser.add_argument_group('tcn', 'configuration for TCN')
    tcn_cmd.add_argument('--kernel_size', type=int)
    tcn_cmd.add_argument('--channels', type=str, help='Channel sizes for TCN layers (comma separated)')
    # lstm
    lstm_cmd = parser.add_argument_group('lstm', 'configuration for LSTM')
    lstm_cmd.add_argument('--layers', type=int)
    lstm_cmd.add_argument('--layer_size', type=int)
    # ODELSTM specific arguments
    odelstm_cmd = parser.add_argument_group('odelstm', 'configuration for ODELSTM')
    odelstm_cmd.add_argument('--hidden_size', type=int, help='Hidden size of the ODE-LSTM model')
    odelstm_cmd.add_argument('--use_frequencies', type=str, help='List of frequencies to use (comma-separated)')
    odelstm_cmd.add_argument('--predict_last_n', type=str,
                             help='Number of steps to predict at each frequency (comma-separated)')
    odelstm_cmd.add_argument('--seq_length', type=str, help='Sequence lengths for each frequency (comma-separated)')
    odelstm_cmd.add_argument('--ode_random_freq_lower_bound', type=str,
                             help='Random frequency lower bound for ODELSTM')
    odelstm_cmd.add_argument('--ode_num_unfolds', type=int, default=1,
                             help='Number of unfolds for ODE solver')
    odelstm_cmd.add_argument('--ode_method', type=str, choices=['euler', 'heun', 'rk4'],
                             default='euler', help='ODE solver method')

    mode = parser.add_subparsers(title='mode', dest='mode', help='Operation: [train] train model, [test] evaluate model')
    mode.required = True
    # train
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str)
    train_cmd.add_argument('--val_list', type=str)
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int)
    train_cmd.add_argument('--save_interval', type=int)
    train_cmd.add_argument('--lr', '--learning_rate', type=float)
    # test
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_path', type=str, default=None)
    test_cmd.add_argument('--test_list', type=str, default=None)
    test_cmd.add_argument('--model_path', type=str, default=None)
    test_cmd.add_argument('--fast_test', action='store_true')
    test_cmd.add_argument('--show_plot', action='store_true')

    '''
    Extra arguments
    Set True: use_scheduler, 
              quite (no output on stdout), 
              force_lr (force lr when a model is loaded from continue_from)
    float:  dropout, 
            max_ori_error (err. threshold for priority grv in degrees)
            max_velocity_norm (filter outliers in training) 
    '''

    args, unknown_args = parser.parse_known_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)

    print(args, kwargs)
    if args.mode == 'train':
        train(args, **kwargs)
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        test(args, **kwargs)
