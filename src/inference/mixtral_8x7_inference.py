import os
import sys
from tqdm import tqdm
from copy import deepcopy

from utils.utils import DataUtils
from utils.opentom_utils import OpenToMUtils

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria, 
    StoppingCriteriaList,
)


def convert_to_mixtral_prompt(chatgpt_prompt: list) -> torch.Tensor:

    tokenizer = AutoTokenizer.from_pretrained(
        'mistralai/Mixtral-8x7B-Instruct-v0.1',
        cache_dir='/scratch/prj/inf_llmcache/hf_cache/',
    )

    BOS_TOKEN, EOS_TOKEN = tokenizer.bos_token_id, tokenizer.eos_token_id
    B_INST, E_INST = "[INST] ", "[/INST]"

    mixtral_prompt = torch.tensor([[BOS_TOKEN]])

    for idx, content_dict in enumerate(chatgpt_prompt):

        # user instruction
        if idx % 2 == 0:
            mixtral_prompt = torch.concat((
                mixtral_prompt, 
                tokenizer.encode(
                    f"{B_INST} {content_dict['content']} {E_INST}",
                    add_special_tokens=False,
                    return_tensors='pt'
                )
            ), dim=-1)

        # assistant response
        else:
            temp_prompt = tokenizer.encode(
                f"{content_dict['content']}",
                add_special_tokens=False,
                return_tensors='pt',
            )
            mixtral_prompt = torch.concat((mixtral_prompt, temp_prompt, torch.tensor([[EOS_TOKEN]])), dim=-1)

    return mixtral_prompt


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stops = [], encounters: int =1, device: str = 'cuda'):
        super().__init__()
        self.stops = [stop.to(device) for stop in stops]
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        last_token = input_ids[0][-1]
        for stop in self.stops:
            if self.tokenizer.decode(stop) == self.tokenizer.decode(last_token):
                return True
        return False


class MixtralInference():

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_NAME = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
    CACHE_DIR = '/scratch/prj/inf_llmcache/hf_cache/'
    generation_config = None
    datautils = DataUtils()
    dirname = os.path.dirname(__file__)

    def init_model(self):

        self.model = AutoModelForCausalLM.from_pretrained(
            self.MODEL_NAME,
            cache_dir=self.CACHE_DIR,
            device_map="auto", 
            torch_dtype = torch.float16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME, 
            cache_dir=self.CACHE_DIR,
        )

    def _create_stopping_criteria(self, stop_tokens: list):
        stop_token_ids = [self.tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_tokens]
        self.hf_stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(self.tokenizer, stops=stop_token_ids, device=self.DEVICE)])


    @torch.no_grad()
    def inference(self, prompt: str, config: dict = {}, stop_tokens: list = []) -> str:

        if isinstance(prompt, str):
            model_inputs = self.tokenizer.encode(prompt.strip(), return_tensors="pt", add_special_tokens=True).to(self.DEVICE)
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
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            else:
                output = self.model.generate(
                    model_inputs,
                    generation_config=self.generation_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        else:
            if stop_tokens:
                output = self.model.generate(
                    model_inputs,
                    stopping_criteria=self.hf_stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            else:
                output = self.model.generate(
                    model_inputs,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

        output = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output

    @classmethod
    def _set_generation_config(cls, config: dict) -> GenerationConfig:
        generation_config = GenerationConfig.from_pretrained(
            cls.MODEL_NAME,
            cache_dir=cls.CACHE_DIR,
            **config,
        )

        cls.generation_config = generation_config
