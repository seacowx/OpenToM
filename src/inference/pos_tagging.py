'''
This script is used to do POS tagging on ToMi dataset

Returns:
    the tagged ToMi dataset will be saved in ../data/tagged_data/ as json files
'''

import json
import argparse
from tqdm import tqdm
from typing import Tuple
from utils import DataUtils

import torch
from transformers import AutoModelWithHeads, RobertaTokenizer


def process_entry(entry: str) -> list:
    '''
    process_entry format entry for pos tagging

    Args:
        entry (str): input sentences

    Returns:
        list: list of processed sentences
    '''
    entry_list = entry.split('\n')
    entry_list = [e[2:] for e in entry_list if e != '']
    return entry_list


def init_model():
    '''
    init_model initialize model for pos tagging
    '''
    global tokenizer, model

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base", cache_dir='/scratch/users/k23035472/hf_cache')
    model = AutoModelWithHeads.from_pretrained("roberta-base", cache_dir='/scratch/users/k23035472/hf_cache')

    adapter_name = model.load_adapter("AdapterHub/roberta-base-pf-conll2003_pos", source="hf")
    model.set_active_adapters(adapter_name)

    model.to(DEVICE)


@torch.no_grad()
def pos_tagging(model, tokenizer, sent: str) -> dict:
    '''
    pos_tagging use pretrained model to do pos tagging

    Args:
        sent (str): input sentence

    Returns:
        dict: model output
    '''
    inputs = tokenizer(sent, return_tensors="pt").to(DEVICE)
    outputs = model(**inputs)

    return outputs


def label_to_tag(sent: str, outputs: list) -> dict:
    '''
    label_to_tag convert model output to POS tag

    Args:
        outputs (list): model output

    Returns:
        dict: POS tag
    '''
    outputs = outputs['logits'].argmax(-1).squeeze().tolist()[1:-1]
    outputs = [tag_map[str(o)] for o in outputs]
    sent_tokenized = tokenizer.encode(sent, add_special_tokens=False)
    sent_tokenized = tokenizer.convert_ids_to_tokens(sent_tokenized)

    assert len(outputs) == len(sent_tokenized)

    sent_tagged_list = []
    for i in range(len(sent_tokenized)):
        sent_tagged_list.append({ 
            'token': sent_tokenized[i], 
            'tag': outputs[i] 
        }) 

    out = {
        'sent': sent,
        'sent_tagged': sent_tagged_list,
    }

    return out


def combine_token_pieces(sent_tagged: list) -> list:
    '''
    combine_token_pieces combine tokenized word pieces

    Args:
        sent_tagged (list): tokenized sentence

    Returns:
        list: tokenized sentence with word pieces combined
    '''
    i, j = 0, 0

    while j < len(sent_tagged):
        if sent_tagged[j][0].isupper() and sent_tagged[j][0] != 'Ġ':
            sent_tagged[i] = sent_tagged[j]
            i += 1
            j += 1

        elif sent_tagged[j][0] == 'Ġ':
            sent_tagged[i] = sent_tagged[j][1:]
            i += 1
            j += 1

        else:
            sent_tagged[i-1] += sent_tagged[j]
            j += 1

    sent_tagged = list(set(sent_tagged[:i]))
    return sent_tagged


def get_args():
    '''
    get_args get arguments from command line

    Returns:
        _parser_: parser with arguments
    '''
    parser = argparse.ArgumentParser(description='EDA')
    parser.add_argument(
        '--data_path', type=str, default='../data/ToMi/data/train.txt',
    )
    return parser.parse_args()


def main():
    global DEVICE, tag_map
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Running on {DEVICE}')
    tag_map = json.load(open('../data/tag_map.json', 'r'))

    args = get_args()
    datautils = DataUtils()

    # check if the data is already tagged
    out_name = ''
    if 'train' in args.data_path:
        out_name = 'train' 
    elif 'val' in args.data_path:
        out_name = 'val'
    else:
        out_name = 'test'

    save_path = f'../data/tagged_data/tmoi_{out_name}_tagged.json'
    datautils.check_file_existence(save_path)

    # initialize model and tokenizer
    init_model()
    data = datautils.load_tomi(args.data_path)

    for key, val in tqdm(data.items()):
        cur_content, cur_question, cur_answer = val['content'], val['question'], val['answer']
        cur_content = process_entry(cur_content)
        cur_question = process_entry(cur_question)

        content_ent_list = []
        for sent in cur_content:
            outputs = pos_tagging(model, tokenizer, sent)
            outputs = label_to_tag(sent, outputs)

            cur_ents = [e['token'] for e in outputs['sent_tagged'] if 'NN' in e['tag']]
            content_ent_list += cur_ents

        question_ent_list = []
        for sent in cur_question:
            outputs = pos_tagging(model, tokenizer, sent)
            outputs = label_to_tag(sent, outputs)

            cur_ents = [e['token'] for e in outputs['sent_tagged'] if 'NN' in e['tag']]
            question_ent_list += cur_ents

        content_ent_list = combine_token_pieces(content_ent_list)
        question_ent_list = combine_token_pieces(question_ent_list)

        data[key]['content_ent'] = content_ent_list
        data[key]['question_ent'] = question_ent_list
        data[key]['answer'] = cur_answer.split()[0]

    datautils.save_json(data, save_path) 


if __name__ == '__main__':
    main()
