import sys, os
import random
import argparse
from tqdm import tqdm

import torch
import numpy as np

from utils.utils import DataUtils
from utils.opentom_utils import OpenToMUtils 
from utils.build_prompt import build_prompt
from utils.load_baseline_model import LoadBaselineModel


def result_io(result_dict, sampled_keys, long_narrative_prefix, cached_narrative_dict):

    if args.high_level_attitude:
        if args.cot:
            datautils.save_json(
                result_dict, 
                f'../data/results/attitude/{args.model}{long_narrative_prefix}_cot.json'
            )
            datautils.save_json(
                sampled_keys, 
                f'../data/results/attitude/sampled_keys/{args.model}{long_narrative_prefix}_cot.json'
            )
        else:
            datautils.save_json(
                result_dict, 
                f'../data/results/attitude/{args.model}{long_narrative_prefix}.json'
            )
            datautils.save_json(
                sampled_keys, 
                f'../data/results/attitude/sampled_keys/{args.model}{long_narrative_prefix}.json'
            )

    else:
        if 'all' in args.question_type:

            if args.cot:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_cot.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}_cot.json'
                )

            elif args.simtom:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_simtom.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}_simtom.json'
                )
            else:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}.json'
                )

            if cached_narrative_dict:
                datautils.save_json(cached_narrative_dict, f'../data/results/{args.model}{long_narrative_prefix}_simtom_narrative.json')

        else:
            if args.cot:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_cot_fg.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}_cot_fg.json'
                )
            if args.selfask:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_selfask.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}_selfask.json'
                )
            elif args.simtom:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_simtom_fg.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}_simtom_fg.json'
                )
            else:
                datautils.save_json(
                    result_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_fg.json'
                )
                datautils.save_json(
                    sampled_keys, 
                    f'../data/results/sampled_keys/{args.model}{long_narrative_prefix}_fg.json'
                )

            if cached_narrative_dict:
                datautils.save_json(
                    cached_narrative_dict, 
                    f'../data/results/{args.model}{long_narrative_prefix}_fg_simtom_narrative.json'
                )


def sample_entries(meta_data: dict, batch_size: int = 50):

    keys = list(meta_data.keys())
    random.shuffle(keys)

    for i in range(0, len(keys), batch_size):
        yield keys[i:i+batch_size]


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_questions(key: str, valid_q_types: list, all_questions: dict):
    cur_questions = {}
    counter = 0
    if 'all' not in args.question_type:
        for q_type in args.question_type:
            assert q_type in valid_q_types, f'Invalid question type: {q_type}'

            new_questions = all_questions[q_type][key]

            # NOTE: randomly remove 2 fullness questions to make the total question regarding fullness to be 2
            if 'multihop' in q_type:
                fullness_idx = [idx for idx, q in enumerate(new_questions) if 'fullness' in q['question']]
                fullness_sampled_idx = np.random.choice(fullness_idx, 2, replace=False)
                new_questions = [q for idx, q in enumerate(new_questions) if idx not in fullness_sampled_idx]

            for q_entry in new_questions:
                cur_questions[str(counter)] = q_entry
                counter += 1

    else:
        for q_type in valid_q_types:

            new_questions = all_questions[q_type][key]

            # NOTE: randomly remove 2 fullness questions to make the total question regarding fullness to be 2
            if 'multihop' in q_type:
                fullness_idx = [idx for idx, q in enumerate(new_questions) if 'fullness' in q['question']]
                fullness_sampled_idx = np.random.choice(fullness_idx, 2, replace=False)
                new_questions = [q for idx, q in enumerate(new_questions) if idx not in fullness_sampled_idx]

            for q_entry in new_questions:
                cur_questions[str(counter)] = q_entry
                counter += 1

    return cur_questions


def get_result(
    key: str, 
    val: dict, 
    prev_val_idx: str,
    valid_q_types: list, 
    all_questions: dict, 
    model, 
    model_info: dict, 
    cached_narrative_dict: dict,
    config_dict: dict = None,
) -> dict:

    if args.long_narrative:
        cur_narrative = val['long_narrative']
    else:
        cur_narrative = val['narrative']

    cur_questions = sample_questions(key, valid_q_types, all_questions)

    mover, affected_char, original_place, move_to_place, eoi = opentom_utils.get_info(val) 
    ac_preference = val['preferences']['observer']

    for question_idx, question_dict in cur_questions.items():
        question_content = question_dict['question']
        question_content_tokens = question_content.replace("'s", '').split()

        cur_prompt, coi = build_prompt(
            question_content,
            question_content_tokens,
            question_dict,
            mover,
            affected_char,
            original_place,
            move_to_place,
            eoi,
            cur_narrative,
            model_info,
            args.cot,
            args.simtom,
            args.selfask,
            high_level_attitude=args.high_level_attitude,
            ac_preference=ac_preference,
        )

        if 'llama' in args.model:
            llama_prompt_converter = model_info['prompt_converter']
            cur_message = model_info["chatgpt_prefix"] + [{"role": "user", "content": cur_prompt}]
            cur_message = llama_prompt_converter(cur_message)
            config_dict = model_info['config']
            result = model.inference(cur_message, config_dict, stop_tokens=['\n', '\n\n'])
            result = result.split(cur_prompt)[-1].lower()

            print(result)

            cur_questions[question_idx]['prediction'] = result

        elif 'mixtral' in args.model:

            config_dict = model_info['config']
            mixtral_prompt_converter = model_info['prompt_converter']

            # NOTE: Generate character-centric narrative for simtom
            if isinstance(cur_prompt, list):
                cur_message = []
                for p in cur_prompt:
                    cur_message.append([{"role": "user", "content": p}]) 
            else:
                cur_message = [{"role": "user", "content": cur_prompt}]

            if args.simtom and prev_val_idx != key:

                print('Generating character-centric narrative for simtom...')
                for coi, m in zip([mover, affected_char], cur_message):
                    cur_input = mixtral_prompt_converter(m)
                    result = model.inference(cur_input, config_dict, stop_tokens=['\n', '\n\n'])
                    result = result.split('[/INST]')[-1].lower()

                    cached_narrative_dict[key][coi] = result
                    cached_narrative_dict[key][f'{coi}_prompt'] = m[-1]['content']

                prev_val_idx = key

            if not args.simtom:
                cur_input = mixtral_prompt_converter(cur_message)
                result = model.inference(cur_input, config_dict, stop_tokens=['\n', '\n\n'])
                result = result.split('[/INST]')[-1].lower()

            else:
                cur_prompt, coi = build_prompt(
                    question_content,
                    question_content_tokens,
                    question_dict,
                    mover,
                    affected_char,
                    original_place,
                    move_to_place,
                    eoi,
                    '{narrative}',
                    model_info,
                    args.cot,
                    args.simtom,
                    args.selfask,
                    simtom_stage=2,
                )

                # NOTE: roll back to original narrative if coi is narrator (omniscent view)
                if 'narrator' in coi:
                    cur_prompt = cur_prompt.replace('{narrative}', cached_narrative_dict[key]['narrative'])
                    cur_message = [{"role": "user", "content": cur_prompt}]

                elif (cur_coi_narrative := cached_narrative_dict[key][coi]) != 'Invalid Request Error Occurred':
                    cur_prompt = cur_prompt.replace('{narrative}', '').strip()
                    cur_message = [{"role": "user", "content": cached_narrative_dict[key][f"{coi}_prompt"]}] + \
                        [{"role": "system", "content": cur_coi_narrative}] + \
                        [{"role": "user", "content": cur_prompt}]

                cur_input = mixtral_prompt_converter(cur_message)
                result = model.inference(cur_input, config_dict, stop_tokens=['\n', '\n\n'])
                result = result.split('[/INST]')[-1].lower()

            cur_questions[question_idx]['prediction'] = result

        elif 'gpt' in args.model:

            # NOTE: Generate character-centric narrative for simtom
            if isinstance(cur_prompt, list):
                cur_message = []
                for p in cur_prompt:
                    cur_message.append(model_info["chatgpt_prefix"] + [{"role": "user", "content": p}]) 
            else:
                cur_message = model_info["chatgpt_prefix"] + [{"role": "user", "content": cur_prompt}]

            if args.simtom and prev_val_idx != key:

                print('Generating character-centric narrative for simtom...')
                for coi, m in zip([mover, affected_char], cur_message):
                    try:
                        result = model.inference(m, temperature=0)
                    except:
                        result = 'Invalid Request Error Occurred'

                    cached_narrative_dict[key][coi] = result
                    cached_narrative_dict[key][f'{coi}_prompt'] = m[-1]['content']

                prev_val_idx = key

            if not args.simtom:

                try:
                    result = model.inference(cur_message, temperature=0)
                except:
                    result = 'Invalid Request Error Occurred'

                cur_questions[question_idx]['prediction'] = result

            else:
                cur_prompt, coi = build_prompt(
                    question_content,
                    question_content_tokens,
                    question_dict,
                    mover,
                    affected_char,
                    original_place,
                    move_to_place,
                    eoi,
                    '{narrative}',
                    model_info,
                    args.cot,
                    args.simtom,
                    args.selfask,
                    simtom_stage=2,
                )

                # NOTE: roll back to original narrative if coi is narrator (omniscent view)
                if 'narrator' in coi:
                    cur_prompt = cur_prompt.replace('{narrative}', cached_narrative_dict[key]['narrative'])
                    cur_message = model_info["chatgpt_prefix"] + [{"role": "user", "content": cur_prompt}]

                elif (cur_coi_narrative := cached_narrative_dict[key][coi]) != 'Invalid Request Error Occurred':
                    cur_prompt = cur_prompt.replace('{narrative}', '').strip()
                    cur_message = model_info["chatgpt_prefix"] + \
                        [{"role": "user", "content": cached_narrative_dict[key][f"{coi}_prompt"]}] + \
                        [{"role": "system", "content": cur_coi_narrative}] + \
                        [{"role": "user", "content": cur_prompt}]

                try:
                    result = model.inference(cur_message, temperature=0)
                except:
                    result = "Invalid Request Error Occurred."

                print(result)

                cur_questions[question_idx]['prediction'] = result

    return cur_questions


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, default='llama-13b', help='choose between llama, gpt, and mixtral'
    )
    parser.add_argument(
        '--question_type', type=str, default='all', nargs='+', help='choose which question type of evaluate, use "all" for all questions'
    )
    parser.add_argument(
        '--config_path', type=str, default=None, help='path to the config file for the model'
    )
    parser.add_argument(
        '--cot', action='store_true', default=False, help='whether to use chain-of-thought prompting'
    )
    parser.add_argument(
        '--simtom', action='store_true', default=False, help='whether to use simtom prompting'
    )
    parser.add_argument(
        '--selfask', action='store_true', default=False, help='whether to use self-ask prompting'
    )
    parser.add_argument(
        '-lg', '--location_granularity', type=str, default='coarse', help='choose between fine and coarse location granularity'
    )
    parser.add_argument(
        '--long_narrative', action='store_true', default=False, help='whether to evaluate on long narrative'
    )
    parser.add_argument(
        '--num_batch', type=int, default=5, help='number of batches to evaluate'
    )
    parser.add_argument(
        '--seed', type=int, default=42, help='random seed'
    )
    parser.add_argument(
        '--batch_size', type=int, default=50, help='batch size for evaluation'
    )
    parser.add_argument(
        '--high_level_attitude', action='store_true', default=False, help='whether to evaluate attitude using high-level abstraction'
    )
    return parser.parse_args()


def main():
    set_seed()
    global args, datautils, opentom_utils
    args = get_args()
    datautils = DataUtils()
    opentom_utils = OpenToMUtils()
    load_baseline_model = LoadBaselineModel(args.config_path, args.cot, args.simtom, args.selfask)
    cur_batch_ids = None

    if args.high_level_attitude:
        valid_q_types = ['attitude']

        # NOTE: ensure the same set of questions are evaluated 
        cur_batch_ids = datautils.load_json(f'../data/baseline_results/v2_results/sampled_keys/{args.model}.json')

    else:
        if args.location_granularity == 'fine':
            valid_q_types = [
                'location_fg_fo', 
                'location_fg_so', 
                'multihop_fo', 
                'multihop_so',
                'attitude'
            ]

            # NOTE: ensure the same set of questions are evaluated 
            cur_batch_ids = datautils.load_json(f'../data/baseline_results/v2_results/sampled_keys/{args.model}.json')

        elif args.location_granularity == 'coarse':
            valid_q_types = [
                'location_cg_fo', 
                'location_cg_so', 
                'multihop_fo', 
                'multihop_so',
                'attitude'
            ]

        elif args.location_granularity == 'both':
            valid_q_types = [
                'location_cg_fo', 
                'location_fg_fo', 
                'location_cg_so', 
                'location_fg_so', 
                'multihop_fo', 
                'multihop_so',
                'attitude'
            ]

    if args.long_narrative:
        meta_data = datautils.load_json('../data/opentom_data/meta_data_long.json')
    else:
        meta_data = datautils.load_json('../data/opentom_data/meta_data.json')

    model, model_info = load_baseline_model.init_model(args.model)

    # NOTE: initialize narrative cache for simtom
    if args.simtom:

        cached_narrative_dict = {}
        for key, val in meta_data.items():

            # load long narrative if specified
            if args.long_narrative:
                cached_narrative_dict[key] = {
                    'narrative': val['gpt_long_narrative']
                }
            else:
                cached_narrative_dict[key] = {
                    'narrative': val['gpt_narrative']
                }

    else:
        cached_narrative_dict = None

    all_questions = {}
    for q_type in valid_q_types:
        new_questions = datautils.load_json(os.path.join('../data/opentom_data/', f'{q_type}.json'))
        all_questions[q_type] = new_questions

    stop_tokens = []
    config_dict = None
    prev_val_idx = None
    result_dict = {f'batch-{i+1}': {} for i in range(args.num_batch)}

    sampled_keys = []
    counter = 0

    while counter < args.num_batch:

        # NOTE: use cached keys if eval on long narrative for result comparison
        if not cur_batch_ids:
            cur_batch = sample_entries(meta_data, batch_size=args.batch_size)
        else:
            cur_batch = cur_batch_ids

        for key_list in cur_batch:

            pbar = tqdm(total=50, position=0, leave=False, desc=f'Batch {counter+1}')

            if counter == args.num_batch:
                break
            counter += 1

            sampled_keys.append(key_list)

            for key in key_list:
                val = meta_data[key]

                cur_result = get_result(
                    key, 
                    val, 
                    prev_val_idx, 
                    valid_q_types, 
                    all_questions, 
                    model, 
                    model_info, 
                    cached_narrative_dict,
                    config_dict,
                )
                
                result_dict[f'batch-{counter}'][key] = cur_result
                pbar.update(1)

        long_narrative_prefix = ''
        if args.long_narrative:
            long_narrative_prefix = '_long'

    result_io(result_dict, sampled_keys, long_narrative_prefix, cached_narrative_dict)

if __name__ == '__main__':
    main()
