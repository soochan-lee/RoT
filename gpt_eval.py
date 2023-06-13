#!/usr/bin/env python
# coding: utf-8

import json
import os
import os.path as path
import random
import time
from collections import namedtuple
from datetime import datetime
from glob import glob
from functools import partial
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import openai
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer
from ratelimit import limits, sleep_and_retry

from data import PROBLEM
from data.problem import build_vocab, ProblemSet, collate_by_len
from eval import Evaluator
from gpt_fine_tune import GPTDataGenerator

openai.api_key = os.getenv('OPENAI_API_KEY')



gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def count_tokens(gpt_data):
    count = 0
    for datum in tqdm(gpt_data):
        count += len(gpt2_tokenizer(datum['prompt'])['input_ids'])
        count += len(gpt2_tokenizer(datum['completion'])['input_ids'])
    return count


def bill(count, training: bool, verbose=False):
    if training:
        prices = {
            'Ada': 0.0004,
            'Babbage': 0.0006,
            'Curie': 0.003,
            'Davinci': 0.03
        }
    else:
        prices = {
            'Ada': 0.0016,
            'Babbage': 0.0024,
            'Curie': 0.012,
            'Davinci': 0.12
        }
    costs = {
        model: count / 1000 * unit_price
        for model, unit_price in prices.items()
    }
    if verbose:
        print(f'{count:,} tokens')
        for model, cost in costs.items():
            print(f'{model}: ${cost:.2f}')
    return costs


def openai_to_dict(obj):
    if isinstance(obj, dict):
        return {
            key: openai_to_dict(value)
            for key, value in obj.items()
        }
    elif isinstance(obj, list):
        return [openai_to_dict(elem) for elem in obj]
    else:
        return obj


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])




Experiment = namedtuple('Experiment',
                        ['prob_name', 'prob_size', 'model', 'paradigm'])


def get_exp_name(exp):
    return f'{exp.prob_name}-{exp.prob_size}-{exp.paradigm}'


def get_exp_dir(exp):
    return f'gpt3/{exp.prob_name}-{exp.prob_size}-{exp.paradigm}'


def get_exp_episode(exp):
    episode_path = f'episodes/{exp.prob_name}-{exp.prob_size}.yaml'
    with open(episode_path, 'r') as f:
        episode = yaml.load(f, Loader=yaml.FullLoader)
    return episode


def get_exp_vocab(exp):
    episode = get_exp_episode(exp)
    prob_classes = [PROBLEM[prob_spec['name']] for prob_spec in episode]
    return build_vocab(prob_classes, paradigm=exp.paradigm)


@sleep_and_retry
@limits(calls=1, period=0.025)
def save_inference(args, model_id):
    example, save_path = args
    if path.isfile(save_path):
        return True
    max_tokens = len(gpt2_tokenizer(example['completion'])['input_ids']) + 1
    for retry in range(10):
        try:
            result = openai.Completion.create(
                model=model_id,
                prompt=example['prompt'],
                max_tokens=max_tokens,
                temperature=0
            )
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2)
            return True
        except openai.error.RateLimitError as e:
            # print(e)
            time.sleep(5)

    print('Maximum retry exceed. Failed to evaluate an example.')
    return False


def evaluate(exp):
    processes = 32
    exp_dir = get_exp_dir(exp)
    infer_dir = path.join(exp_dir, 'inferences')
    eval_result_path = path.join(exp_dir, 'eval_result.yaml')
    if path.isfile(eval_result_path):
        # Already done
        return

    evaluator_path = path.join(exp_dir, 'evaluator.pt')
    fine_tune_complete_path = path.join(exp_dir, 'fine_tune_complete.yaml')
    if not path.isfile(evaluator_path) or not path.isfile(
            fine_tune_complete_path):
        print('Dependencies not met.')
        return

    dummy_config = {'eval_length_budget': 1000, 'num_workers': processes}
    generator = GPTDataGenerator(exp)
    evaluator = Evaluator(dummy_config, exp.paradigm, vocab=get_exp_vocab(exp))
    evaluator.load_state_dict(torch.load(evaluator_path))
    with open(fine_tune_complete_path, 'r') as f:
        fine_tune_complete = yaml.load(f, Loader=yaml.FullLoader)
    model_id = fine_tune_complete['fine_tuned_model']

    os.makedirs(infer_dir, mode=0o700, exist_ok=True)

    eval_data = []
    infer_args = []
    skip_count = 0
    for i, (prob_cls, args) in enumerate(tqdm(evaluator.sorted_probs)):
        x, y, _ = prob_cls.solve(args, paradigm=generator.paradigm)
        datum = generator.xy_to_gpt_data(generator.vocab(x), generator.vocab(y))
        eval_data.append(datum)
        for j, example in enumerate(datum):
            save_path = path.join(infer_dir, f'{i}-{j}.json')
            if path.isfile(save_path):
                # Already done
                skip_count += 1
                continue
            infer_args.append((example, save_path))

    print(
        f'Calling API for {len(infer_args)} examples, skipping already finished {skip_count} examples.')
    print(f'Model ID: {model_id}')
    with ThreadPoolExecutor(max_workers=16) as pool:
        successes = list(tqdm(
            pool.map(partial(save_inference, model_id=model_id), infer_args),
            total=len(infer_args)))
    if not all(successes):
        print('Found failed API calls. Retry evaluation later...')
        return

    # Aggregate results
    corrects = []
    wrongs = []
    for i, datum in enumerate(tqdm(eval_data)):
        correct = True
        for j, example in enumerate(datum):
            result_path = path.join(infer_dir, f'{i}-{j}.json')
            with open(result_path, 'r') as f:
                result = json.load(f)
            if not result['choices'][0]['text'].startswith(
                    example['completion']):
                correct = False
                wrongs.append((
                    example['prompt'],
                    example['completion'],
                    result['choices'][0]['text'],
                    result['choices'][0]['finish_reason']
                ))
        corrects.append(correct)

    node_eval = {
        prob: correct
        for prob, correct in zip(evaluator.sorted_probs, corrects)
    }
    correct_deep, correct_shallow, prob_total = evaluator.aggregate_eval(
        node_eval)
    with open(eval_result_path, 'w') as f:
        eval_result = {
            'correct': sum(correct_deep.values()),
            'total': sum(prob_total.values()),
        }
        print(eval_result)
        yaml.dump(eval_result, f)
    print(f'Evaluation result written to {eval_result_path}')


def main():
    while True:
        try:
            for exp_dir in glob('gpt3/*'):
                fine_tune_complete_path = path.join(exp_dir, 'fine_tune_complete.yaml')
                eval_result_path = path.join(exp_dir, 'eval_result.yaml')
                if path.isfile(fine_tune_complete_path) and not path.isfile(eval_result_path):
                    prob_name, prob_size, paradigm = path.basename(exp_dir).split('-')
                    exp = Experiment(prob_name, prob_size, 'gpt3', paradigm)
                    evaluate(exp)
        except Exception as e:
            print(e)
        time.sleep(30)


if __name__ == '__main__':
    main()
