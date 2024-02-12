import os
import numpy as np
from glob import glob

from .utils import DataUtils


class OpenToMUtils:


    def get_info(self, val: dict) -> tuple[str, str, str, str, str]:
        """
        function to get the characters, objects and locations involved in the ToMi narrative

        Args:
            val: a ToMi narrative entry

        Returns:
            mover: the character who moves the object 
            affected_char: the character who is potentially affected by the movement 
            original_place: the original location of the object 
            move_to_place: the destination location of the object 
            eoi: the object 
        """

        if 'plot_info' in val.keys():
            mover, affected_char, eoi, original_place, move_to_place = val['plot_info'].values()

        else:
            cur_content = val['plot']
            cur_questions = val['questions']
            all_context_ent = val['context_ent']

            eoi, coi = self.get_entity_of_interest(cur_questions, all_context_ent)

            content_sents = cur_content.split('\n')

            mover = ''
            move_to_place = ''
            original_place = ''
            flag = 1

            for sent in content_sents:

                if flag and eoi in sent:
                    sent_tokens = sent.replace('.', '').split() 
                    original_place = ''
                    for token in sent_tokens:
                        if token in all_context_ent and token != eoi and token[0].islower():
                            original_place += token
                            flag = 0

                if 'move' in sent:
                    sent_tokens = sent.replace('.', '').split()
                    mover = []
                    move_to_place = ''
                    for token in sent_tokens:
                        if token[0].isupper():
                            mover.append(token)

                    move_to_place = sent.split('to the')[-1].strip()

            # sanity check: there should be only one mover in the context
            mover = list(set(mover))
            if len(mover) > 1:
                raise ValueError('More than one mover found in the context.')

            mover = mover[0]
            # the mover should be in the characters of interest
            assert mover in coi, 'Mover not in characters of interest.'
            affected_char = [c for c in coi if c != mover]
            # there should only be one character affected in the context
            assert len(affected_char) == 1
            affected_char = affected_char[0]

            # there must be a place affected in the narrative
            assert move_to_place != '', 'No place affected found in the context.'

            # there must be an original place in the narrative
            # assert original_place != '', 'No original place found in the context.'

        return mover, affected_char, original_place, move_to_place, eoi


    @staticmethod
    def get_entity_of_interest(questions: dict, all_ents: list) -> tuple:
        """
        get_entity_of_interest funtion to get entity of interest in the questions. Returns the most common entity of interest.

        Args:
            questions: list of questions
            all_ents: list of all entities in the context

        Returns:
            str: object of interest
            list: characters of interest
        """
        eoi = None
        coi = []
        for ent in all_ents:
            if ent[0].islower() and ent in questions['1']['question']:
                eoi = ent

            for question in questions.values():
                if ent[0].isupper() and ent in question['question']:
                    coi.append(ent)

        if not eoi:
            raise ValueError('No entity of interest found in the context.')

        coi = list(set(coi))

        return (eoi, coi)


    @staticmethod 
    def cache_tom_data(data: dict, cache_path: str, model: str, **kwargs) -> None:
        datautils = DataUtils()
        existing_files = glob(os.path.join(cache_path, '*.json'))

        post_fix = ''
        for key, val in kwargs.items():
            if isinstance(val, str) and 'shot' in val:
                post_fix += '_' + f'{str(val)}_shot'
            elif val:
                post_fix += '_' + key.strip()

        existing_files = [file for file in existing_files if post_fix in file]
        existing_ids = [f.split('_')[-1].split('.')[0] for f in existing_files]
        existing_ids = [int(ele) for ele in existing_ids if ele.isnumeric()]

        new_id = np.random.randint(1000000, 9999999)
        while new_id in existing_ids:
            new_id = np.random.randint(1000000, 9999999)

        if model:
            new_fname = f'tomi_{model}' + post_fix + '_' + str(new_id) + '.json'
        else:
            new_fname = f'tomi' + post_fix + '_' + str(new_id) + '.json'

        datautils.save_json(data, os.path.join(cache_path, new_fname))

