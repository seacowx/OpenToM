from typing import Final
from utils.utils import DataUtils
from inference.gpt_inference import GPTInference
from inference.cosmo_inference import CosmoAgent
from inference.sentiment_classifier import SentimentClassifier
from inference.llama_inference import LlamaInference, convert_to_llama_prompt
from inference.mixtral_8x7_inference import MixtralInference, convert_to_mixtral_prompt

class LoadBaselineModel():

    def __init__(
        self, 
        config_path: str, 
        cot: bool=False, 
        simtom: bool=False, 
        selfask: bool=False,
    ) -> None:

        self.datautils = DataUtils()
        self.config_path = config_path
        self.cot = cot
        self.simtom = simtom
        self.selfask = selfask

    def _load_model(self, user_model: str):

        model = None
        model_info = {}
        model_info['model_name'] = user_model

        if 'llama' in user_model:

            model = LlamaInference()

            if '-' not in user_model:
                model.set_model(model_name='13b', chat=True)
            else:
                model_size = user_model.split('-')[-1]
                model.set_model(model_name=model_size, chat=True)

            if self.config_path:
                config_dict = self.datautils.load_yaml(self.config_path)
                model_info['config'] = config_dict
            else:
                model_info['config'] = None

            attitude_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_attitude.txt')
            location_fg_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_location_fg.txt')
            location_cg_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_location_cg.txt')
            multihop_fullness_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_multihop_fullness.txt')
            multihop_accessibility_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_multihop_accessibility.txt')
            preference_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_preference.txt')
            intention_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_intention.txt')

            # add prompt templates to model_info
            model_info['attitude_prompt_template'] = attitude_prompt_template
            model_info['location_cg_prompt_template'] = location_cg_prompt_template
            model_info['location_fg_prompt_template'] = location_fg_prompt_template
            model_info['multihop_fullness_prompt_template'] = multihop_fullness_prompt_template
            model_info['multihop_accessibility_prompt_template'] = multihop_accessibility_prompt_template
            model_info['preference_prompt_template'] = preference_prompt_template
            model_info['intention_prompt_template'] = intention_prompt_template

            # add function to convert chatgpt prompt to llama prompt
            model_info['prompt_converter'] = convert_to_llama_prompt
            chatgpt_prefix = [
                {"role": "system", "content": "You are an expert in modeling other's mental state."},
            ]
            model_info['chatgpt_prefix'] = chatgpt_prefix

            if self.cot:
                model_info['cot_postfix'] = "Reason step by step before answering. Write the answer in the end."

            if self.selfask:
                model_info['cot_postfix'] = "Break the original question into sub-questions. Explicitly state the follow-up questions, and the answers to the follow-up questions. Aggregate the answers to the follow-up questions and write the answer in the end as \"Final Answer: [answer]\"."

            if self.simtom:
                model_info['simtom_s1'] = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_simtom_s1.txt')
                model_info['simtom_s2'] = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_simtom_s2.txt')

            # initialize llama model and tokenizer
            model.init_model()

        elif 'mixtral' in user_model:

            model = MixtralInference()

            if self.config_path:
                config_dict = self.datautils.load_yaml(self.config_path)
                model_info['config'] = config_dict
            else:
                model_info['config'] = None

            attitude_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_attitude.txt')
            location_fg_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_location_fg.txt')
            location_cg_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_location_cg.txt')
            multihop_fullness_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_multihop_fullness.txt')
            multihop_accessibility_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_multihop_accessibility.txt')
            preference_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_preference.txt')
            intention_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_intention.txt')

            # add prompt templates to model_info
            model_info['attitude_prompt_template'] = attitude_prompt_template
            model_info['location_cg_prompt_template'] = location_cg_prompt_template
            model_info['location_fg_prompt_template'] = location_fg_prompt_template
            model_info['multihop_fullness_prompt_template'] = multihop_fullness_prompt_template
            model_info['multihop_accessibility_prompt_template'] = multihop_accessibility_prompt_template
            model_info['preference_prompt_template'] = preference_prompt_template
            model_info['intention_prompt_template'] = intention_prompt_template

            # add function to convert chatgpt prompt to llama prompt
            model_info['prompt_converter'] = convert_to_mixtral_prompt

            if self.cot:
                model_info['cot_postfix'] = "Reason step by step before answering. Write the answer in the end."

            if self.selfask:
                model_info['cot_postfix'] = "Break the original question into sub-questions. Explicitly state the follow-up questions, and the answers to the follow-up questions. Aggregate the answers to the follow-up questions and write the answer in the end as \"Final Answer: [answer]\"."

            if self.simtom:
                model_info['simtom_s1'] = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_simtom_s1.txt')
                model_info['simtom_s2'] = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_simtom_s2.txt')

            # initialize llama model and tokenizer
            model.init_model()

        elif 'gpt' in user_model:
            model = GPTInference()

            if '4' in user_model:
                model.set_openai_config('/Users/seacow/hainiu_openai_gpt4.config')
            else:
                model.set_openai_config('/Users/seacow/hainiu_openai_chatgpt.config')

            chatgpt_prefix = [
                {"role": "system", "content": "You are an expert in modeling other's mental state."},
            ]
            model_info['chatgpt_prefix'] = chatgpt_prefix

            attitude_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_attitude.txt')
            location_fg_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_location_fg.txt')
            location_cg_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_location_cg.txt')
            multihop_fullness_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_multihop_fullness.txt')
            multihop_accessibility_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_multihop_accessibility.txt')
            preference_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_preference.txt')
            intention_prompt_template = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_intention.txt')

            # add prompt templates to model_info
            model_info['attitude_prompt_template'] = attitude_prompt_template
            model_info['location_cg_prompt_template'] = location_cg_prompt_template
            model_info['location_fg_prompt_template'] = location_fg_prompt_template
            model_info['multihop_fullness_prompt_template'] = multihop_fullness_prompt_template
            model_info['multihop_accessibility_prompt_template'] = multihop_accessibility_prompt_template
            model_info['preference_prompt_template'] = preference_prompt_template
            model_info['intention_prompt_template'] = intention_prompt_template

            if self.cot:
                model_info['cot_postfix'] = "Reason step by step before answering. Write the answer in the end."

            if self.selfask:
                model_info['cot_postfix'] = "Break the original question into sub-questions. Explicitly state the follow-up questions, and the answers to the follow-up questions. Aggregate the answers to the follow-up questions and write the answer in the end as \"Final Answer: [answer]\"."

            if self.simtom:
                model_info['simtom_s1'] = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_simtom_s1.txt')
                model_info['simtom_s2'] = self.datautils.load_txt('./prompts/chatgpt_opentom_prompts/chatgpt_simtom_s2.txt')

        elif 'cosmo' in user_model:
            model = CosmoAgent()
            model_info['sentiment_model'] = SentimentClassifier()

        return model, model_info

    def _sanity_check(self, user_model: str, model_info: dict):

        if 'llama' in user_model:
            assert 'config' in model_info.keys(), 'config not found in model_info'

            if self.cot:
                assert 'cot_postfix' in model_info.keys(), 'cot_postfix not found in model_info'

            assert 'attitude_prompt_template' in model_info.keys(), 'attitude_prompt_template not found in model_info'
            assert 'location_cg_prompt_template' in model_info.keys(), 'entity_state_prompt_template not found in model_info'
            assert 'location_fg_prompt_template' in model_info.keys(), 'entity_state_prompt_template not found in model_info'
            assert 'multihop_fullness_prompt_template' in model_info.keys(), 'multihop_fullness_prompt_template not found in model_info'
            assert 'multihop_accessibility_prompt_template' in model_info.keys(), 'multihop_accessibility_prompt_template not found in model_info'
            assert 'preference_prompt_template' in model_info.keys(), 'preference_prompt_template not found in model_info'
            assert 'intention_prompt_template' in model_info.keys(), 'intention_prompt_template not found in model_info'

        elif 'mixtral' in user_model:
            assert 'config' in model_info.keys(), 'config not found in model_info'

            if self.cot:
                assert 'cot_postfix' in model_info.keys(), 'cot_postfix not found in model_info'

            assert 'attitude_prompt_template' in model_info.keys(), 'attitude_prompt_template not found in model_info'
            assert 'location_cg_prompt_template' in model_info.keys(), 'entity_state_prompt_template not found in model_info'
            assert 'location_fg_prompt_template' in model_info.keys(), 'entity_state_prompt_template not found in model_info'
            assert 'multihop_fullness_prompt_template' in model_info.keys(), 'multihop_fullness_prompt_template not found in model_info'
            assert 'multihop_accessibility_prompt_template' in model_info.keys(), 'multihop_accessibility_prompt_template not found in model_info'
            assert 'preference_prompt_template' in model_info.keys(), 'preference_prompt_template not found in model_info'
            assert 'intention_prompt_template' in model_info.keys(), 'intention_prompt_template not found in model_info'

        elif 'gpt' in user_model:

            if self.cot:
                assert 'cot_postfix' in model_info.keys(), 'cot_postfix not found in model_info'

            assert 'chatgpt_prefix' in model_info.keys(), 'chatgpt_prefix not found in model_info'
            assert 'attitude_prompt_template' in model_info.keys(), 'attitude_prompt_template not found in model_info'
            assert 'location_cg_prompt_template' in model_info.keys(), 'entity_state_prompt_template not found in model_info'
            assert 'location_fg_prompt_template' in model_info.keys(), 'entity_state_prompt_template not found in model_info'
            assert 'multihop_fullness_prompt_template' in model_info.keys(), 'multihop_fullness_prompt_template not found in model_info'
            assert 'multihop_accessibility_prompt_template' in model_info.keys(), 'multihop_accessibility_prompt_template not found in model_info'
            assert 'preference_prompt_template' in model_info.keys(), 'preference_prompt_template not found in model_info'
            assert 'intention_prompt_template' in model_info.keys(), 'intention_prompt_template not found in model_info'

        elif 'cosmo' in user_model:
            assert 'sentiment_model' in model_info.keys(), 'sentiment_model not found in model_info'

    def init_model(self, model_name: str):

        model, model_info = None, {}
        model, model_info = self._load_model(model_name)

        self._sanity_check(model_name, model_info)

        return model, model_info
