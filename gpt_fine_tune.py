#!/usr/bin/env python
# coding: utf-8



import json
import os
import os.path as path
import random
import time
from collections import namedtuple
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import openai
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
from transformers import GPT2Tokenizer

from data import PROBLEM
from data.problem import build_vocab, ProblemSet, collate_by_len
from eval import Evaluator

openai.api_key = os.getenv('OPENAI_API_KEY')



gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def count_tokens(gpt_data):
    count = 0
    for datum in gpt_data:
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




class GPTDataGenerator:
    def __init__(self, exp):
        self.episode = get_exp_episode(exp)
        self.paradigm = exp.paradigm

        # Build vocab
        self.vocab = get_exp_vocab(exp)
        self.itos = self.vocab.get_itos()
        self.gpt_itos = [
            self.to_gpt_token(token)
            for token in self.vocab.get_itos()
        ]

        # Build problems
        self.problems = [
            PROBLEM[prob_spec['name']](exp.paradigm, self.vocab,
                                       prob_spec['config'])
            for prob_spec in self.episode
        ]
        print(', '.join([f'{problem}' for problem in self.problems]))
        self.problem_set = ProblemSet(self.problems, paradigm=exp.paradigm,
                                      vocab=self.vocab)

        # Train loader
        self.train_loader = self.problem_set.get_data_loader(
            batch_size=1, num_workers=0,
            collate_fn=collate_by_len)

    def generate(self, num, sample_ratio=1.0):
        train_loader_iter = iter(self.train_loader)
        data = []
        while len(data) < num:
            (x, y, l), = next(train_loader_iter)
            examples = self.xy_to_gpt_data(x, y)
            for example in examples:
                if len(data) >= num:
                    break
                if random.random() < sample_ratio:
                    data.append(example)
        random.shuffle(data)
        return data

    def generate_at(self, num, path, sample_ratio=1.0):
        data = self.generate(num, sample_ratio=sample_ratio)
        pd.DataFrame(data).to_json(path, orient='records', lines=True)

    @staticmethod
    def to_gpt_token(token):
        if token == '÷':
            token = '/'
        elif token == '×':
            token = 'by'
        elif token.startswith('<'):
            token = token[1:-1].lower()
        return ' ' + token
        # return ' ' + token[1:-1].lower() if token.startswith('<') else token

    def decode_tokens(self, x):
        if isinstance(x, torch.Tensor):
            x = x.view(-1)
        return [self.itos[token] for token in x]

    def decode_tokens_gpt(self, x):
        if isinstance(x, torch.Tensor):
            x = x.view(-1)
        return [self.gpt_itos[token] for token in x]

    def xy_to_gpt_data(self, x, y):
        x_dec = self.decode_tokens_gpt(x)
        y_dec = self.decode_tokens_gpt(y)

        prompts = []
        completions = []
        prompt_end = 0
        for i, (x_t, y_t) in enumerate(zip(x_dec, y_dec)):
            match x_t, y_t:
                case (_, ' think') | (_, ' stop'):
                    completions.append(y_dec[prompt_end:i + 1])
                case (_, ' go') | (_, ' tail') | (' stop', _) | (' =', _):
                    prompts.append(x_dec[:i + 1])
                    prompt_end = i

        prompts = [''.join(prompt) for prompt in prompts]
        completions = [''.join(completion) for completion in completions]
        assert len(prompts) == len(completions)
        data = [
            {
                'prompt': prompt,
                'completion': completion if completion.startswith(
                    ' ') else ' ' + completion
            }
            for prompt, completion in zip(prompts, completions)
        ]
        return data



experiments = [
    Experiment(prob_name='Add', prob_size=32, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Add', prob_size=32, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Add', prob_size=48, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Add', prob_size=48, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Div', prob_size=16, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Div', prob_size=16, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Div', prob_size=8, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Div', prob_size=8, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Knapsack', prob_size=4, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Knapsack', prob_size=4, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Knapsack', prob_size=6, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Knapsack', prob_size=6, model='gpt3', paradigm='wt'),
    Experiment(prob_name='LCS', prob_size=16, model='gpt3', paradigm='rot'),
    Experiment(prob_name='LCS', prob_size=16, model='gpt3', paradigm='wt'),
    Experiment(prob_name='LCS', prob_size=20, model='gpt3', paradigm='rot'),
    Experiment(prob_name='LCS', prob_size=20, model='gpt3', paradigm='wt'),
    Experiment(prob_name='LCS', prob_size=24, model='gpt3', paradigm='rot'),
    Experiment(prob_name='LCS', prob_size=24, model='gpt3', paradigm='wt'),
    Experiment(prob_name='LPS', prob_size=24, model='gpt3', paradigm='rot'),
    Experiment(prob_name='LPS', prob_size=24, model='gpt3', paradigm='wt'),
    Experiment(prob_name='LPS', prob_size=32, model='gpt3', paradigm='rot'),
    Experiment(prob_name='LPS', prob_size=32, model='gpt3', paradigm='wt'),
    Experiment(prob_name='LPS', prob_size=40, model='gpt3', paradigm='rot'),
    Experiment(prob_name='LPS', prob_size=40, model='gpt3', paradigm='wt'),
    Experiment(prob_name='MCM', prob_size=3, model='gpt3', paradigm='rot'),
    Experiment(prob_name='MCM', prob_size=3, model='gpt3', paradigm='wt'),
    Experiment(prob_name='MCM', prob_size=4, model='gpt3', paradigm='rot'),
    Experiment(prob_name='MCM', prob_size=4, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Mul', prob_size=16, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Mul', prob_size=16, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Mul', prob_size=8, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Mul', prob_size=8, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Sub', prob_size=32, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Sub', prob_size=32, model='gpt3', paradigm='wt'),
    Experiment(prob_name='Sub', prob_size=48, model='gpt3', paradigm='rot'),
    Experiment(prob_name='Sub', prob_size=48, model='gpt3', paradigm='wt'),
]

# Generate Training Data

processes = 32
train_examples = 256 * 10_000
chunk_size = 2560
num_jobs = train_examples // chunk_size
eval_probs = 1000
dummy_config = {'eval_length_budget': 1000, 'num_workers': processes}
with Pool(processes=processes) as pool:
    for exp in experiments:
        exp_dir = get_exp_dir(exp)
        episode_path = f'episodes/{exp.prob_name}-{exp.prob_size}.yaml'
        os.makedirs(exp_dir, 0o700, exist_ok=True)

        # Create training data
        train_data_path = path.join(exp_dir, 'train.jsonl')
        train_info_path = path.join(exp_dir, 'train_info.yaml')
        generator = GPTDataGenerator(exp)
        if not path.isfile(train_info_path):
            print(f'Creating {train_data_path}')
            gen_result = list(tqdm(pool.imap(
                partial(generator.generate, sample_ratio=0.3),
                [chunk_size] * num_jobs
            ), total=num_jobs))
            print('Computing training cost...')
            train_tokens = sum(list(tqdm(pool.imap(count_tokens, gen_result),
                                         total=len(gen_result))))
            gen_concat = []
            for data in gen_result:
                gen_concat.extend(data)
            random.shuffle(gen_concat)
            pd.DataFrame(gen_concat).to_json(train_data_path, orient='records',
                                             lines=True)
            with open(train_info_path, 'w') as f:
                yaml.dump({
                    'tokens': train_tokens,
                    'prices': bill(train_tokens, training=True)
                }, f)

        # Create evaluation data
        evaluator = Evaluator(dummy_config, generator.paradigm, generator.vocab)
        evaluator_path = path.join(exp_dir, 'evaluator.pt')
        eval_info_path = path.join(exp_dir, 'eval_info.yaml')
        if not path.isfile(evaluator_path):
            print(f'Creating evaluator at {evaluator_path}')
            top_probs = []
            for problem in generator.problems:
                for args in sorted(problem.get_unique_args(eval_probs)):
                    top_probs.append((problem.__class__, args))
            evaluator.add_probs(top_probs)
            evaluator.update()
            torch.save(evaluator.state_dict(), evaluator_path)

            print(f'Computing evaluation cost...')
            eval_data = []
            for prob_cls, args in tqdm(evaluator.sorted_probs):
                x, y, _ = prob_cls.solve(args, paradigm=generator.paradigm)
                eval_data.append(generator.xy_to_gpt_data(generator.vocab(x),
                                                          generator.vocab(y)))

            eval_tokens = sum(list(
                tqdm(pool.imap(count_tokens, eval_data), total=len(eval_data))))
            with open(eval_info_path, 'w') as f:
                yaml.dump({
                    'tokens': eval_tokens,
                    'prices': bill(eval_tokens, training=False)
                }, f)


def upload_training_data(exp):
    exp_dir = get_exp_dir(exp)
    exp_name = get_exp_name(exp)
    train_data_path = path.join(exp_dir, 'train.jsonl')
    train_info_path = path.join(exp_dir, 'train_info.yaml')
    train_uploaded_path = path.join(exp_dir, 'train_uploaded.yaml')
    if path.isfile(train_data_path) and path.isfile(
            train_info_path) and not path.isfile(train_uploaded_path):
        print(f'Uploading {train_data_path}')
        with open(train_data_path, 'r') as f:
            uploaded_file = openai.File.create(
                file=open(train_data_path),
                purpose='fine-tune',
                user_provided_filename=exp_name
            )
        print(f'Uploaded as {uploaded_file["id"]}')
        with open(train_uploaded_path, 'w') as f:
            yaml.dump(openai_to_dict(uploaded_file), f)


def delete_completed_training_data():
    for fine_tune in openai.FineTune.list()['data']:
        for training_file in fine_tune['training_files']:
            if fine_tune['status'] == 'succeeded' and training_file[
                'status'] != 'deleted':
                result = openai.File.delete(training_file['id'])
                if result['deleted']:
                    print(
                        f'Training file of {training_file["filename"]} deleted')


def request_fine_tuning(exp):
    exp_dir = get_exp_dir(exp)
    exp_name = get_exp_name(exp)
    train_uploaded_path = path.join(exp_dir, 'train_uploaded.yaml')
    fine_tune_path = path.join(exp_dir, 'fine_tune.yaml')
    if path.isfile(train_uploaded_path) and not path.isfile(fine_tune_path):
        print(f'Request fine-tuning of {exp_name}')
        with open(train_uploaded_path, 'r') as f:
            training_file_id = yaml.load(f, Loader=yaml.FullLoader)['id']
        fine_tune = openai.FineTune.create(
            training_file=training_file_id,
            model='ada',
            n_epochs=1,
            batch_size=256,
            prompt_loss_weight=0,
            suffix=exp_name
        )
        with open(fine_tune_path, 'w') as f:
            yaml.dump(openai_to_dict(fine_tune), f)


def check_fine_tune(exp):
    exp_dir = get_exp_dir(exp)
    fine_tune_path = path.join(exp_dir, 'fine_tune.yaml')
    if not path.isfile(fine_tune_path):
        print('No fine-tuning record found. Request fine-tuning first.')
        return

    with open(fine_tune_path, 'r') as f:
        fine_tune_id = yaml.load(f, Loader=yaml.FullLoader)['id']
    fine_tune = openai.FineTune.retrieve(id=fine_tune_id)
    if fine_tune['status'] == 'succeeded':
        print(f'Fine-tune completed: {fine_tune["id"]}')
        with open(path.join(exp_dir, 'fine_tune_complete.yaml'), 'w') as f:
            yaml.dump(openai_to_dict(fine_tune), f)
    return fine_tune


def save_inference(args, model_id):
    example, save_path = args
    if path.isfile(save_path):
        return
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
            break
        except openai.error.RateLimitError:
            time.sleep(5)
    else:
        print('Maximum retry exceed. Failed to evaluate an example.')


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
    with Pool(16) as pool:
        list(tqdm(
            pool.imap(partial(save_inference, model_id=model_id), infer_args),
            total=len(infer_args)))

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


def progress():
    fine_tune_queue = []
    for exp in experiments:
        exp_dir = get_exp_dir(exp)
        exp_name = get_exp_name(exp)

        train_info_path = path.join(exp_dir, 'train_info.yaml')
        train_uploaded_path = path.join(exp_dir, 'train_uploaded.yaml')
        fine_tune_path = path.join(exp_dir, 'fine_tune.yaml')
        fine_tune_complete_path = path.join(exp_dir, 'fine_tune_complete.yaml')

        if path.isfile(train_info_path) and not path.isfile(
                train_uploaded_path) and len(fine_tune_queue) < 2:
            print(datetime.now())
            delete_completed_training_data()
            upload_training_data(exp)
        if path.isfile(train_uploaded_path) and not path.isfile(fine_tune_path):
            print(datetime.now())
            request_fine_tuning(exp)
        if path.isfile(fine_tune_path) and not path.isfile(
                fine_tune_complete_path):
            fine_tune = check_fine_tune(exp)
            if fine_tune['status'] == 'succeeded':
                print(datetime.now())
                print(f'{exp_name} fine-tuning completed')
            else:
                fine_tune_queue.append(exp)


def main():
    while True:
        try:
            progress()
        except Exception as e:
            print(e)
        time.sleep(60)


if __name__ == '__main__':
    main()
