import re
import sys, os
import openai
import backoff
import warnings
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from transformers import GPT2Tokenizer

from utils.utils import DataUtils
from utils.opentom_utils import OpenToMUtils


class GPTInference():

    datautils = DataUtils()

    @staticmethod 
    def est_token_size(prompt_examples: list) -> int:
        """
        est_token_size to estimate the average number of tokens in a prompt

        Args:
            prompt_examples: list of prompt examples

        Returns:
            estimated token number in one prompt
        """

        if not isinstance(prompt_examples, list):
            prompt_examples = [prompt_examples]

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        total = []
        for prompt in prompt_examples:
            tokenized_prompt = tokenizer(prompt).input_ids
            total.append(len(tokenized_prompt))

        return round(np.mean(total))

    @staticmethod
    def est_cost(est_prompt_size: int, est_output_size: int, model_name: str, sample_size: int, ncalls: int = 1) -> float:
        '''
        est_cost function to estimate the cost of an example prompt

        Args:
            est_prompt_size (str): estimated prompt length (in tokens)
            est_output_size (str): estimated output length (in tokens)
            model_name (str): name of the model
            sample_size (int): number of samples to generate
            ncalls (int, optional): number of calls to the API. Defaults to 1.

        Returns:
            float: estimated cost of the example prompt
        '''
        input_cost_dict = {
            '3.5': 0.0015,
            '4': 0.03
        }

        output_cost_dict = {
            '3.5': 0.002,
            '4': 0.06
        }

        if '4' in model_name:
            input_cost = input_cost_dict['4']
            output_cost = output_cost_dict['4']
        else:
            input_cost = input_cost_dict['3.5']
            output_cost = output_cost_dict['3.5']

        prompt_cost = est_prompt_size * sample_size * ncalls / 1000 * input_cost
        output_cost = est_output_size * sample_size * ncalls / 1000 * output_cost
        return prompt_cost + output_cost

    def set_openai_config(self, config_path: str) -> None:
        '''
        set_openai_key function to set the openai key

        Args:
            config_path (str): path to the Azure OpenAI configuration
        '''
        try:
            self.openai_config = self.datautils.load_yaml(config_path)
            openai.api_type = self.openai_config['api_type']
            openai.api_base = self.openai_config['api_base']
            openai.api_version = self.openai_config['api_version']
            openai.api_key = self.openai_config['api_key']
        except:
            raise ValueError('Invalid openai configuration file')

    @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, openai.error.APIError, openai.error.Timeout, openai.error.ServiceUnavailableError, openai.error.APIConnectionError))
    def inference(self, prompt, temperature=1.0, max_tokens=1024):
        ret = openai.ChatCompletion.create(
            engine=self.openai_config['deploy_name'],
            messages=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )

        try:
            gen_text = dict(ret["choices"][0]["message"])["content"]
            return gen_text
        except:
            return ''

    def _make_narrative_prompt(self, val: dict, prompt: list, opentom_utils: OpenToMUtils) -> tuple[list, dict]:

        cur_content = val['plot'] 
        cur_preferences = val['preferences']

        personality_statement = ''
        mover_preference_statement = ''
        affected_char_preference_statement = ''

        personality_statement = val['personality']
        sentiment_statement = val['sentiment_statement']
        cur_intention = val['intention']
        destination = val['new_location']

        mover, affected_char, _, _, eoi = opentom_utils.get_tomi_info(val)

        new_plot, present_flag = self._modify_plot(cur_content, eoi, mover, affected_char)
        val['observed'] = present_flag

        mover_preference_statement = cur_preferences['mv']
        affected_char_preference_statement = cur_preferences['ac']

        cur_plot = ''
        cur_plot += f'Paragraph 1: {mover_preference_statement} {affected_char_preference_statement}\n'

        # get plot content up to the movement
        cur_content_wo_movement = ''
        for sent in new_plot.split('\n'):
            if 'move' in sent:
                break
            cur_content_wo_movement += sent + ' '

        cur_plot += f'Paragraph 2: {cur_content_wo_movement}\n'

        if 'inconsiderate' in personality_statement:
            personal_preference = cur_preferences['mv']
            main_event = f'{personality_statement} {personal_preference} Therefore, {mover} moved the {eoi} to {destination} in order to {cur_intention}.'

        elif 'considerate' in personality_statement:
            main_event = f'{personality_statement} {sentiment_statement} Therefore, {mover} moved the {eoi} to {destination} in order to {cur_intention}.'

        elif 'negativistic' in personality_statement:
            if cur_preferences['mv_ac_sentiment']:
                rationale = cur_preferences['mv_ac_sentiment']
            else:
                rationale = cur_preferences['ac']

            sentiment_statement = sentiment_statement.replace('.', '')
            main_event = f'{personality_statement} {sentiment_statement} because {rationale} Therefore, {mover} moved the {eoi} to {destination} in order to {cur_intention}.'

        else:
            ac_preference = cur_preferences['ac']
            mv_ac_preference = cur_preferences['mv_ac_sentiment']
            
            if mv_ac_preference:
                main_event = f'{personality_statement} {mv_ac_preference} Hence, {sentiment_statement} {mover} moved the {eoi} to {destination} in order to {cur_intention}.'
            else:
                main_event = f'{personality_statement} {ac_preference} Hence, {sentiment_statement} {mover} moved the {eoi} to {destination} in order to {cur_intention}.'

        # add main event to plot
        if present_flag:
            cur_plot += f'Paragraph 3: {main_event} {affected_char} witnessed {mover}\'s action.\n'
        else:
            cur_plot += f'Paragraph 3: {main_event} {affected_char} did not witness {mover}\'s action.\n'


        cur_prompt = deepcopy(prompt)
        cur_prompt[-1]['content'] = prompt[-1]['content'].replace('{plot}', cur_plot)

        val['plot'] = new_plot

        return cur_prompt, val

    def _modify_plot(self, plot: str, eoi: str, mover: str, affected_char: str) -> str:
        """
        modify the original plot to make the story flow more natrual

        Args:
            plot: current plot

        Returns:
            modified plot
        """
        plot_list = plot.split('\n')
        mover_cur_place, affected_char_cur_place = '', ''

        # NOTE: First scan the plot and check the final location of the mover and the affected character
        present_flag = False
        for idx, sent in enumerate(plot_list):

            if mover in sent and 'move' not in sent:
                mover_cur_place = sent.split()[-1].replace('.', '')

            if affected_char in sent and 'exit' not in sent:
                affected_char_cur_place = sent.split()[-1].replace('.', '')

            if affected_char in sent and 'exit' in sent:
                affected_char_cur_place = 'absent'

            if eoi in sent and 'move' not in sent:
                plot_list[idx] = f'Both {mover} and {affected_char} noticed that ' + sent.replace('.', '').lower() + f' in the {mover_cur_place}.'

            if 'move' in sent:
                present_flag = mover_cur_place == affected_char_cur_place

                if present_flag:

                    # add possibility that the observer may not observed the event even if he/she is present
                    if np.random.rand() < 0.85:
                        plot_list[idx] = sent.replace('.', ' ') +  f'and {affected_char} witnessed the action.'
                    else:
                        plot_list[idx] = sent.replace('.', ' ') +  f'but {affected_char} did not witness the action.'

                else:
                    plot_list[idx] = sent.replace('.', ' ') +  f'and {affected_char} did not witness the event.'

        plot = '\n'.join(plot_list)

        return plot, present_flag

    def modify_narrative(self, dataset: dict) -> dict:
        """

        Funtion that utilizes GPT model to compose narrative based on the original ToMi dataset

        Args:
            dataset: the original ToMi dataset

        Returns:
            dataset: the modified ToMi dataset   
        """

        opentom_utils = ToMiUtils()

        # load prompt template
        prompt = self.datautils.load_jsonl('../prompts/chatgpt_narrative.jsonl')

        for key, val in tqdm(dataset.items()): 

            cur_prompt, val = self._make_narrative_prompt(val, prompt, opentom_utils)

            result = self.inference(cur_prompt)

            dataset[key] = val
            dataset[key]['prompt'] = cur_prompt
            dataset[key]['gpt_narrative'] = result

        # avg_input_token = gpt_inference.est_token_size(prompt_list)
        # est_cost = gpt_inference.est_cost(avg_input_token, 300, '3.5', 20, 1)
        # print(f'Estimated cost: ${est_cost}.')
        return dataset


    # NOTE: This function has been replaced by modify_places. 

    # def get_entity_locations(self, entity: str, sentiment: str) -> list:
    #     sentiment = 'like' if sentiment == 'positive' else 'hate'
    #
    #     cur_prompt = deepcopy(self.prompt)
    #     cur_prompt[-1]['content'] = cur_prompt[-1]['content'].replace('{eoi}', entity).replace('{sentiment}', sentiment)
    #
    #     output = self.inference(cur_prompt)
    #     output = [ele.replace('-', '').strip() for ele in output.split('\n')]
    #     return output

    def _generate_intention_destination(self, cur_prompt: list, eoi: str, mover: str, cur_true_sentiment: str, tolerance: int) -> tuple[str, str]:
        flag = False
        result = ''
        retry_counter = 0
        while not flag:
            result = self.inference(cur_prompt)

            # check the integrity of the generated intentions
            for idx, sent in enumerate(result.strip().split('\n')):

                if sent[0] != str(idx + 1):
                    flag = False
                    break

                if 'move' not in sent:
                    flag = False
                    break
                else:
                    flag = True

            if not flag:
                retry_counter += 1
                warnings.warn(f'Invalid intention detected. Re-generating intention...')

            if retry_counter == tolerance:
                return '', ''

        # based on the proposed intention and action, extract the best one 
        cur_prompt += [{"role": "assistant", "content": result}]
        cur_prompt += [{
            "role": "user", 
            "content": f"Of the potential intentions, which one do you think is {cur_true_sentiment}? Answer with the original sentence. Do not add any additional words."
        }]

        flag = False

        pred_eoi, best_destination, best_intention = '', '', ''
        retry_counter = 0 
        while not flag:
            best_intention_action = self.inference(cur_prompt)
            groups = re.match(rf'{mover.capitalize()} would move the (.*) to (.*) in order to (.*).', best_intention_action)
            if groups:
                pred_eoi, best_destination, best_intention = groups.groups()
                if pred_eoi.strip().lower() == eoi.strip().lower():
                    flag = True

                if best_destination.strip() == '{location}':
                    flag = False

            if not flag:
                retry_counter += 1
                warnings.warn('Invalid best intention detected. Re-generating intention...')
            
            if retry_counter == tolerance:
                return '', ''

        return best_intention, best_destination


    def add_intention(self, tomi_data: dict, tolerance: int, key_list: list = []) -> dict:
        """
        This function modifies the personality of the character and generate the intention and action based on the modified personality.

        Args:
            tomi_data: the original ToMi dataset with preference added
            tolerance: the number of times to retry if the generated intention is invalid

        Returns:
            tomi_data: the modified ToMi dataset 
        """

        opentom_utils = ToMiUtils()
        # NOTE: load prompt for paraphrasing the destination 
        destination_paraphrase_prompt = self.datautils.load_jsonl('../prompts/chatgpt_location_paraphrase.jsonl')

        # NOTE: load considreate prompt, which has three cases (both like, both hate, and have diferent opinions)
        considerate_prompt_like = self.datautils.load_jsonl('../prompts/chatgpt_intention_considerate_like.jsonl')
        considerate_prompt_hate = self.datautils.load_jsonl('../prompts/chatgpt_intention_considerate_hate.jsonl')
        considerate_prompt_disagree_like = self.datautils.load_jsonl('../prompts/chatgpt_intention_considerate_disagree_like.jsonl')
        considerate_prompt_disagree_hate = self.datautils.load_jsonl('../prompts/chatgpt_intention_considerate_disagree_hate.jsonl')

        # NOTE: load inconsidreate prompt, which has two cases (the character likes / hates)
        inconsiderate_prompt_like = self.datautils.load_jsonl('../prompts/chatgpt_intention_inconsiderate_like.jsonl')
        inconsiderate_prompt_hate = self.datautils.load_jsonl('../prompts/chatgpt_intention_inconsiderate_hate.jsonl')

        # NOTE: load negavistic prompt, which has two cases (the character likes / hates)
        negativistic_prompt_getrid = self.datautils.load_jsonl('../prompts/chatgpt_intention_negativistic_getrid.jsonl')
        negativistic_prompt_showoff = self.datautils.load_jsonl('../prompts/chatgpt_intention_negativistic_showoff.jsonl')

        num_corrupted = 0
        original_len = len(tomi_data)

        new_tomi_data = deepcopy(tomi_data)
        # add a list to track corrupted entries 
        corrupted_keys = []

        if key_list:
            tomi_data = {k: v for (k, v) in tomi_data.items() if k in key_list}
            message = "Correcting corrupted intentions..."
        else:
            message = "Generating character intentions..."
        
        for idx, (key, val) in enumerate(tqdm(tomi_data.items(), desc=message)):

            cur_plot = val['plot']
            cur_questions = val['questions']
            cur_personality = val['personality']
            cur_sentiment = val['sentiment_statement']
            cur_true_sentiment = val['true_sentiment']

            mover, affected_char, _, destination, eoi = opentom_utils.get_tomi_info(val)

            # use prompt based on the personality of the mover
            cur_prompt = []
            if 'inconsiderate' in cur_personality:
                if 'like' in cur_sentiment:
                    cur_prompt = deepcopy(inconsiderate_prompt_like)
                else:
                    cur_prompt = deepcopy(inconsiderate_prompt_hate)

            elif 'considerate' in cur_personality:
                if 'Although' in cur_sentiment:
                    if 'like' in cur_sentiment.split(',')[-1]:
                        cur_prompt = deepcopy(considerate_prompt_disagree_like)
                    else:
                        cur_prompt = deepcopy(considerate_prompt_disagree_hate)
                else:
                    if 'like' in cur_sentiment:
                        cur_prompt = deepcopy(considerate_prompt_like)
                    else:
                        cur_prompt = deepcopy(considerate_prompt_hate)

            elif 'negativistic' in cur_personality:
                if 'get rid' in cur_sentiment:
                    cur_prompt = deepcopy(negativistic_prompt_getrid)
                else:
                    cur_prompt = deepcopy(negativistic_prompt_showoff)

                # NOTE: reset personality as the word "negativistic" will disturb model generation
                cur_personality = ''

            cur_prompt[-1]['content'] = cur_prompt[-1]['content'].replace('{mover}', mover.capitalize()) \
                                                                .replace('{eoi}', eoi) \
                                                                .replace('{personality_statement}', cur_personality) \
                                                                .replace('{preference_statement}', cur_sentiment) \
                                                                .replace('{affected_char}', affected_char)

            best_intention, best_destination = self._generate_intention_destination(cur_prompt, eoi, mover, cur_true_sentiment, tolerance)

            # paraphrase best_destination to make it short 
            if len(best_destination.split()) >= 5 and mover not in best_destination:
                paraphrase_prompt = deepcopy(destination_paraphrase_prompt)
                paraphrase_prompt[-1]['content'] = paraphrase_prompt[-1]['content'].replace('{best_destination}', best_destination)
                best_destination = self.inference(paraphrase_prompt)


            if best_intention and best_destination:

                new_tomi_data[key]['intention'] = best_intention
                new_tomi_data[key]['new_location'] = best_destination

                # NOTE: replace the old move_to destination with the new one
                new_tomi_data[key]['plot'] = cur_plot.replace(destination, best_destination)

                new_tomi_data[key]['plot_info']['move_to_place'] = best_destination

                # NOTE: Replace the old destination with the new one in the questions
                for question_id, question_dict in cur_questions.items():
                    question = question_dict['question']
                    question = question.replace(destination, best_destination)
                    new_tomi_data[key]['questions'][question_id]['question'] = question

            else:
                corrupted_keys.append(key)
                num_corrupted += 1

        return new_tomi_data, corrupted_keys


# WARNING: Below code is for testing purpose ONLY
def main():
    datautils = DataUtils()
    gpt_inference = GPTInference()

    tomi_data = datautils.load_json('../../data/annotation/tomi_human_attitude_5437627.json')

    tomi_data = gpt_inference.modify_narrative(tomi_data)


if __name__ == '__main__':
    main()
