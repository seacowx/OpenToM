import os
import sys
from tqdm import tqdm
from copy import deepcopy
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer, 
    GenerationConfig,
    StoppingCriteria, 
    StoppingCriteriaList,
)

from utils.utils import DataUtils
from utils.opentom_utils import OpenToMUtils


def convert_to_llama_prompt(chatgpt_prompt: list, model_param: str = '7b') -> str:

    dirname = os.path.dirname(__file__)
    access_token = open(os.path.join(os.path.expanduser('~/hainiu_hf_token.key')), 'r').read().strip()

    tokenizer = LlamaTokenizer.from_pretrained(
        f'meta-llama/Llama-2-{model_param}-chat-hf',
        token=access_token,
        cache_dir='/scratch/prj/inf_llmcache/hf_cache/',
    )

    BOS_TOKEN, EOS_TOKEN = tokenizer.bos_token_id, tokenizer.eos_token_id
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    llama_prompt = tokenizer.encode(
        f"{B_INST} {B_SYS}{chatgpt_prompt[0]['content'].strip()} {E_SYS}",
        add_special_tokens=False,
        return_tensors='pt'
    )

    llama_prompt = torch.concat((torch.tensor([[BOS_TOKEN]]), llama_prompt), dim=-1)

    for idx, content_dict in enumerate(chatgpt_prompt):

        # skip system prompt
        if idx == 0:
            continue

        # user instruction
        elif idx == 1:
            llama_prompt = torch.concat((
                llama_prompt, 
                tokenizer.encode(
                    f"{content_dict['content']} {E_INST}",
                    add_special_tokens=False,
                    return_tensors='pt'
                )
            ), dim=-1)

        # assistant response
        else:
            if content_dict['role'] == 'user':
                temp_prompt = tokenizer.encode(
                    f"{B_INST}{content_dict['content']} {E_INST}",
                    add_special_tokens=False,
                    return_tensors='pt',
                )
                llama_prompt = torch.concat((llama_prompt, torch.tensor([[BOS_TOKEN]]), temp_prompt), dim=-1)

            elif content_dict['role'] == 'assistant':
                temp_prompt = tokenizer.encode(
                    f"{content_dict['content']}",
                    add_special_tokens=False,
                    return_tensors='pt',
                )
                llama_prompt = torch.concat((llama_prompt, temp_prompt, torch.tensor([[EOS_TOKEN]])), dim=-1)

    return llama_prompt


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer: LlamaTokenizer, stops = [], encounters: int =1, device: str = 'cuda'):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False


class LlamaInference():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'meta-llama/Llama-2-7b-chat-hf'
    CACHE_DIR = '/scratch/prj/inf_llmcache/hf_cache/'
    generation_config = None
    datautils = DataUtils()
    dirname = os.path.dirname(__file__)

    # WARNING: Remove when publish to GitHub
    access_token = open(os.path.expanduser('~/hainiu_hf_token.key'), 'r').read().strip()

    def init_model(self):
        # NOTE: Load 70B model with half precision
        if '70' in self.MODEL_NAME:
            self.model = LlamaForCausalLM.from_pretrained(
                self.MODEL_NAME, 
                device_map="auto", 
                token=self.access_token,
                cache_dir=self.CACHE_DIR,
                torch_dtype = torch.float16,
            )
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.MODEL_NAME, 
                device_map="auto", 
                token=self.access_token,
                cache_dir=self.CACHE_DIR,
            )

        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.MODEL_NAME, 
            token=self.access_token,
            cache_dir=self.CACHE_DIR,
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token


    def _create_stopping_criteria(self, stop_tokens: list):
        stop_token_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_tokens]
        self.hf_stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(self.tokenizer, stops=stop_token_ids, device=self.DEVICE)])


    @torch.no_grad()
    def inference(self, prompt: str, config: dict = {}, stop_tokens: list = []) -> str:

        if isinstance(prompt, str):
            model_inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(self.DEVICE)
        else:
            model_inputs = prompt.to(self.DEVICE)

        if stop_tokens:
            self._create_stopping_criteria(stop_tokens)

        if config:
            self._set_generation_config(config)

            if stop_tokens:
                output = self.model.generate(
                    model_inputs,
                    generation_config=self.generation_config,
                    stopping_criteria=self.hf_stopping_criteria,
                )
            else:
                output = self.model.generate(
                    model_inputs,
                    generation_config=self.generation_config,
                )
        else:
            if stop_tokens:
                output = self.model.generate(
                    model_inputs,
                    stopping_criteria=self.hf_stopping_criteria,
                )

            else:
                output = self.model.generate(
                    model_inputs,
                )

        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output

    def modify_narrative(
        self,
        dataset: dict,
        model_size: str = '13b',
        chat: bool = True,
        token_path: str = '../../hainiu_hf_token.key',
        prompt_folder_path: str = '../prompts/',
    ) -> dict:
        '''
        modify_narrative function to modify the narrative by adding preference to the characters

        Args:
            affected_char (str): affected character
            mover (str): mover
            eoi (str): entity of interest
            context (str): context
            model_size (str): size of the model. Choose from 7b, 13b
            chat (bool, optional): whether to use chat model. Defaults to False.

        Returns:
            str: modified narrative
        '''
        self.set_model(model_size, chat)
        self.set_token(token_path)
        self.init_model()

        opentom_utils = OpenToMUtils()

        if chat:
            prompt_name = 'llama_chat_narrative.txt'
        else:
            prompt_name = 'llama_vanilla_narrative.txt'

        prompt_path = os.path.join(prompt_folder_path, prompt_name)
        prompt = open(prompt_path, 'r').read().strip()

        for key, val in tqdm(dataset.items()):
            cur_content = val['plot']
            cur_questions = val['questions']
            all_context_ent = val['context_ent']

            eoi, coi = opentom_utils.get_entity_of_interest(cur_questions, all_context_ent)
            mover, affected_char, _, _ = opentom_utils.get_tomi_info(cur_content, eoi, coi, all_context_ent)

            cur_prompt = prompt.replace('{affected_char}', affected_char) \
                            .replace('{mover}', mover) \
                            .replace('{eoi}', eoi) \
                            .replace('{script}', cur_content)

            output = self.inference(cur_prompt)
            output = output.split(cur_prompt)[-1].strip()

            dataset[key]['llama_narrative'] = output

        return dataset


    def load_location_prompt(self):
        self.prompt = open('../prompts/llama_location.txt', 'r').read().strip()


    # def get_entity_locations(self, entity: str, sentiment: str) -> list:
    #
    #     sentiment = 'likes' if sentiment == 'positive' else 'hates'
    #
    #     config_dict = self.datautils.load_yaml('../configs/llama_location.yml')
    #     cur_prompt = self.prompt.replace('{eoi}', entity).replace('{sentiment}', sentiment)
    #
    #     output = self.inference(cur_prompt, config_dict)
    #     output = output.split(cur_prompt)[-1].strip()
    #     output = output.split('\n\n')[0].split('\n')
    #     output = [ele.split(',') for ele in output]
    #     output = [lst for sublst in output for lst in sublst]
    #     output = [location.replace('-', '').strip() for location in output]
    #     return output


    # WARNING: this function has been deprecated due to poor performance of the Llama2 model
    def add_intention(self, tomi_data: dict, tolerance: int) -> dict:

        raise NotImplementedError
        # opentom_utils = OpenToMUtils()
        #
        # prompt = self.datautils.load_txt('../prompts/llama_motivation.txt')
        # config_dict = self.datautils.load_yaml('../configs/llama_intention.yml')
        #
        # for key, val in tomi_data.items():
        #     cur_plot = val['plot']
        #     cur_questions = val['questions']
        #     all_context_ent = val['context_ent']
        #     cur_event = cur_plot[-1]
        #
        #     eoi, coi = opentom_utils.get_entity_of_interest(cur_questions, all_context_ent)
        #     mover, affected_char, _, _ = opentom_utils.get_tomi_info(cur_plot, eoi, coi, all_context_ent)
        #
        #     sentiment_statement = [sent for sent in cur_plot.split('\n')[:2] if mover.lower() in sent.lower()][0]
        #
        #     cur_prompt = deepcopy(prompt)
        #     cur_prompt = cur_prompt.replace('{mover}', mover.capitalize()) \
        #                             .replace('{eoi}', eoi) \
        #                             .replace('{sentiment_statement}', sentiment_statement) \
        #                             .replace('{move_to_place}', new_destination)
        #
        #     result = self.inference(cur_prompt, config_dict).split(cur_prompt)[-1]
        #
        #     result = result.split('\n')
        #     result = [res for res in result if res.strip()]
        #
        #     tomi_data[key]['intentions'] = result
        #
        # return tomi_data


    @classmethod
    def set_model(cls, model_name: str, chat: bool = True):
        if chat:
            cls.MODEL_NAME = f'meta-llama/Llama-2-{model_name}-chat-hf'
        else:
            cls.MODEL_NAME = f'meta-llama/Llama-2-{model_name}-hf'

    @classmethod
    def set_token(cls, token_path: str):
        cls.access_token = open(token_path, 'r').read().strip()

    @classmethod
    def _set_generation_config(cls, config: dict) -> GenerationConfig:
        generation_config = GenerationConfig.from_pretrained(
            cls.MODEL_NAME,
            cache_dir=cls.CACHE_DIR,
            use_auth_token=cls.access_token,
            **config,
        )

        cls.generation_config = generation_config
