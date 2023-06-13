import os
import os.path as path
import socket
import sys
from argparse import ArgumentParser
from datetime import datetime
from functools import partial
from glob import glob

import torch
import torch.nn as nn
import yaml
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from torch.utils.tensorboard import SummaryWriter

from data import PROBLEM, ProblemSet
from data.problem import build_vocab, collate_by_len
from data.tokenizer import Label
from eval import Evaluator
from models import MODEL
from utils import Timer

sys.setrecursionlimit(100_000)

parser = ArgumentParser()
parser.add_argument('--paradigm', '-p', choices=['wt', 'cot', 'rot'],
                    required=True)
parser.add_argument('--config', '-c')
parser.add_argument('--episode', '-e')
parser.add_argument('--log-dir', '-l')
parser.add_argument('--override', '-o', default='')
parser.add_argument('--resume', action='store_true')

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    print(f'Running on {socket.gethostname()} | {torch.cuda.get_device_name()}')
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    args = parser.parse_args()
    paradigm = args.paradigm

    # Load config
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    episode = yaml.load(open(args.episode), Loader=yaml.FullLoader)
    config['episode'] = episode
    config['paradigm'] = paradigm

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    # Prevent overwriting
    config['log_dir'] = args.log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {args.log_dir} already exists. Skipping...')
        exit(0)

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    episode_save_path = path.join(config['log_dir'], 'episode.yaml')
    yaml.dump(config, open(config_save_path, 'w'))
    yaml.dump(episode, open(episode_save_path, 'w'))
    print('Config & episode saved to {}'.format(config['log_dir']))

    # Build vocab
    prob_classes = [PROBLEM[prob_spec['name']] for prob_spec in episode]
    vocab = build_vocab(prob_classes, paradigm=paradigm)

    # Build model
    model = MODEL[config['model']](config, vocab)
    start_step = 0

    # Training components
    criterion = nn.CrossEntropyLoss(reduction='none')
    writer = SummaryWriter(config['log_dir'], flush_secs=15)
    scaler = torch.cuda.amp.GradScaler(
        init_scale=2. ** 40, growth_interval=1_000_000_000_000)  # constant

    # Resume checkpoint
    if config_exists and args.resume:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            model.optim.load_state_dict(ckpt['optim'])
            model.lr_sched.load_state_dict(ckpt['lr_sched'])
            scaler.load_state_dict(ckpt['grad_scaler'])
            start_step = ckpt['step']
            print(f'Loaded checkpoint at {ckpt_path}')
    model.optim.zero_grad(set_to_none=True)

    # Build problems
    problems = [
        PROBLEM[prob_spec['name']](paradigm, vocab, prob_spec['config'])
        for prob_spec in episode
    ]
    print(', '.join([f'{problem}' for problem in problems]))
    problem_set = ProblemSet(problems, paradigm=paradigm, vocab=vocab)

    # Evaluator
    evaluator = Evaluator(config, paradigm, vocab)
    top_probs = []
    for problem in problems:
        for args in problem.get_unique_args(config['eval_data_size']):
            top_probs.append((problem.__class__, args))
    evaluator.add_probs(top_probs)

    # Evaluate the last checkpoint if needed
    step = start_step
    if step > 0 and step % config['eval_interval'] == 0:
        # Check if the last evaluation succeeded
        summary_path = sorted(glob(
            path.join(config['log_dir'], 'events.out.tfevents.*')))[-1]
        ea = EventAccumulator(summary_path)
        ea.Reload()

        # Evaluate the last checkpoint
        acc_tag = 'accuracy_deep/all'
        if acc_tag not in ea.Tags()['scalars'] or \
                ea.Scalars(acc_tag)[-1].step < step:
            with Timer('Evaluation time: {:.3f}s'):
                torch.cuda.empty_cache()
                evaluation = evaluator.evaluate(model)
                torch.cuda.empty_cache()

            write_summary(evaluation, step, writer)

    # Train loader
    train_loader = problem_set.get_data_loader(
        config['batch_size'], num_workers=config['num_workers'],
        collate_fn=partial(collate_by_len, budget=config['length_budget']))
    train_loader_iter = iter(train_loader)

    # Main training loop
    for step in range(start_step + 1, config['max_train_steps'] + 1):
        splits = next(train_loader_iter)
        train_masks = [
            (label.to(model.device) >= Label.T).type(torch.float)
            for _, _, label in splits
        ]
        train_tokens = sum([mask.sum() for mask in train_masks])
        loss_total = 0.0
        for i, ((x, y, label), train_mask) in \
                enumerate(zip(splits, train_masks)):
            x, y = x.to(model.device), y.to(model.device)

            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=config['amp']):
                output = model(x)
                loss = criterion(
                    output.view(-1, output.shape[-1]), y.view(-1)
                ) * train_mask.view(-1)
                loss = loss.sum() / train_tokens
            loss_total += loss.detach()
            scaler.scale(loss).backward(retain_graph=i < len(splits) - 1)

        scaler.step(model.optim)
        scaler.update()
        model.lr_sched.step()
        model.optim.zero_grad(set_to_none=True)

        if step % config['summary_interval'] == 0:
            writer.add_scalar('loss/train', loss_total, step)
            writer.add_scalar('lr', model.lr_sched.get_last_lr()[0], step)
            writer.add_scalar('splits', len(splits), step)

            # Sequence length summary
            trailing_pads_all = []
            lengths_all = []
            for _, _, label in splits:
                not_pad = (label > Label.PAD).type(torch.int)
                reverse_cumsum = \
                    not_pad + not_pad.sum(0, keepdims=True) \
                    - torch.cumsum(not_pad, 0)
                trailing_pads = (reverse_cumsum == 0).type(torch.float).sum(0)
                lengths = label.shape[0] - trailing_pads
                trailing_pads_all.append(trailing_pads)
                lengths_all.append(lengths)
            trailing_pads = torch.cat(trailing_pads_all)
            lengths = torch.cat(lengths_all)
            writer.add_scalar('trailing_pads/total', trailing_pads.sum(), step)
            writer.add_scalar('trailing_pads/mean', trailing_pads.mean(), step)
            writer.add_scalar('lengths/max', lengths.max(), step)
            writer.add_scalar('lengths/mean', lengths.mean(), step)
            writer.add_scalar('lengths/median', lengths.median(), step)
            writer.add_scalar('lengths/min', lengths.min(), step)
            writer.add_scalar('grad_scaler/scale', scaler.get_scale(), step)

            # Compute remaining time
            now = datetime.now()
            elapsed_time = now - start_time
            elapsed_steps = step - start_step
            total_steps = config['max_train_steps'] - start_step
            est_total = elapsed_time * total_steps / elapsed_steps
            # Remove microseconds for brevity
            elapsed_time = str(elapsed_time).split('.')[0]
            est_total = str(est_total).split('.')[0]
            print(f'\r[Step {step}] [{elapsed_time} / {est_total}] '
                  f'Loss: {loss_total:.8f}', end='')

        if step % config['ckpt_interval'] == 0:
            # Remove old checkpoints
            ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
            for ckpt_path in ckpt_paths[:-4]:
                os.remove(ckpt_path)

            new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step:06}.pt')
            torch.save({
                'step': step,
                'config': config,
                'paradigm': paradigm,
                'model': model.state_dict(),
                'optim': model.optim.state_dict(),
                'lr_sched': model.lr_sched.state_dict(),
                'grad_scaler': scaler.state_dict(),
            }, new_ckpt_path)

        if step % config['eval_interval'] == 0:
            print()
            with Timer('Evaluation time: {:.3f}s'):
                torch.cuda.empty_cache()
                evaluation = evaluator.evaluate(model)
                torch.cuda.empty_cache()

            write_summary(evaluation, step, writer)

            subprob_correct_all = sum(evaluation['subprob_correct'].values())
            subprob_total_all = sum(evaluation['subprob_total'].values())
            if subprob_correct_all == subprob_total_all:
                print('==== Perfect score reached ====')
                break

    writer.flush()
    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')

    with open(path.join(config['log_dir'], 'completed.yaml'), 'a') as f:
        yaml.dump({
            'step': step,
            'end_time': end_time,
        }, f)


def write_summary(evaluation, step, writer):
    # Add scalar summaries
    for metric in [
        'prob_total', 'accuracy_shallow', 'accuracy_deep',
        'subprob_total', 'accuracy_subprob'
    ]:
        for prob_cls, score in evaluation[metric].items():
            writer.add_scalar(f'{metric}/{prob_cls.name}', score, step)

    # Summarize wrong samples
    for prob_cls in evaluation['prob_total']:
        wrong = '\n\n'.join(evaluation['wrong_samples'][prob_cls])
        writer.add_text(
            f'wrong/{prob_cls.name}',
            f'```\n{wrong}\n```', step)

    # Add average accuracies
    for acc_type in ['shallow', 'deep']:
        correct_all = sum(evaluation[f'correct_{acc_type}'].values())
        total_all = sum(evaluation['prob_total'].values())
        writer.add_scalar(
            f'accuracy_{acc_type}/all',
            correct_all / total_all,
            step)

    subprob_correct_all = sum(evaluation['subprob_correct'].values())
    subprob_total_all = sum(evaluation['subprob_total'].values())
    writer.add_scalar(
        'accuracy_subprob/all',
        subprob_correct_all / subprob_total_all,
        step)


if __name__ == '__main__':
    main()
