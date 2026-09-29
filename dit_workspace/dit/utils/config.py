"""Unified YAML and explicit runtime overrides."""
from pathlib import Path
import yaml


def validate_config(config):
    required = ('experiment','data','stft','compression','representation','model','process',
                'sampler','training','validation','metrics','logging','distributed','inference')
    for name in required:
        if name not in config:
            raise ValueError('Missing config section: ' + name)
    data, stft = config['data'], config['stft']
    if data['data_mode'] not in ('paired','on_the_fly') or not data.get('mono', True):
        raise ValueError('Expected paired/on_the_fly mono data')
    if data['sample_rate'] <= 0 or data['segment_seconds'] <= 0 or data['snr_min'] > data['snr_max']:
        raise ValueError('Invalid data rate, duration, or SNR range')
    if not 0 < stft['hop_length'] <= stft['win_length'] <= stft['n_fft']:
        raise ValueError('Require 0 < hop <= win <= n_fft')
    if not stft['center']:
        raise ValueError('This Hann baseline requires center=true for invertible boundaries')
    if config['process']['type'] not in ('score','ddpm','flow'):
        raise ValueError('Unknown process')
    if config['sampler']['num_steps'] < 1:
        raise ValueError('num_steps must be positive')
    if config['training']['batch_size'] < 1 or config['validation']['eval_interval'] < 1:
        raise ValueError('Invalid batch size/eval interval')
    if config['validation']['max_sampling_batches'] < 1:
        raise ValueError('Sampling validation must include at least one batch')
    if config['training'].get('optimizer','adam') != 'adam':
        raise NotImplementedError('Only Adam implemented')
    if config['training'].get('scheduler','reduce_on_plateau') != 'reduce_on_plateau':
        raise NotImplementedError('Only ReduceLROnPlateau implemented')
    if config['training'].get('auxiliary_waveform_weight',0) != 0:
        raise NotImplementedError('Auxiliary waveform loss is not implemented')
    crop = config['compression'].get('crop_frames')
    if crop is not None and crop < 1:
        raise ValueError('crop_frames must be positive')
    inf = config['inference']
    if not inf.get('chunk_seconds') or not 0 <= inf['overlap_seconds'] < inf['chunk_seconds']:
        raise ValueError('Require finite positive chunk_seconds and smaller nonnegative overlap')


def load_config(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        config = yaml.safe_load(stream)
    validate_config(config)
    config['_config_path'] = str(Path(path).resolve())
    return config


def save_config(config, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8') as stream:
        yaml.safe_dump({k:v for k,v in config.items() if not k.startswith('_')}, stream, sort_keys=False)


def add_evaluation_arguments(parser):
    parser.add_argument('--test-manifest')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--process', choices=('score','ddpm','flow'))
    parser.add_argument('--sampler', choices=('pc','ddpm','euler','heun'))
    parser.add_argument('--num-steps', type=int)


def apply_evaluation_arguments(config, args):
    if args.test_manifest:
        config['data']['test_manifest'] = args.test_manifest
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.process and args.process != config['process']['type']:
        raise ValueError('Cannot reinterpret trained weights with another process; select its checkpoint/config')
    if args.sampler:
        config['sampler']['name'] = args.sampler
    if args.num_steps is not None:
        config['sampler']['num_steps'] = args.num_steps
    validate_config(config)
