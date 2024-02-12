import re
import sys
from typing import List
from copy import deepcopy
import numpy as np
sys.path.append('../data_ops')
sys.path.append('../')

import torch

from utils.utils import DataUtils, BaselineLabels 
from utils.opentom_utils import OpenToMUtils


class OpenToMEvaluator():

    def __init__(self) -> None:
        self.datautils = DataUtils()
        self.baseline_labels = BaselineLabels()
        self.opentom_utils = OpenToMUtils()
        # self._init_sentiment_model()

    @staticmethod 
    def remove_determinant(word: str) -> str:
        determinants = ['a', 'an', 'the']
        for det in determinants:
            if word.startswith(det):
                return word[len(det):].strip()
        return word

    @staticmethod
    def compute_lexical_overlap(pred: str, location: str) -> float:
        pred = pred.lower().replace('_', ' ').replace("'s", '')
        location = location.lower().replace('_', ' ').replace("'s", '')
        score = 0 
        pred = pred.replace('.', '').split()
        location = location.split()
        visited_word = []

        for word in pred:
            if word in location and word not in visited_word:
                score += 1
                visited_word.append(word)

        return score / len(location)

    @staticmethod
    def parse_cot_answer(answer: str) -> str:
        # cot typically generate answer in the last sentence or paragraph
        if '\n' in answer:
            answer = answer.split('\n')[-1]
        else:
            answer = answer.split('Therefore')[-1]
        return answer
    
    def check_answer_for_fg_location(self, prediction: str, answer: str, original_place: str, move_to_place: str) -> list:

        # truncate prediction as some of them contain explanations
        answer = self.remove_determinant(answer).lower()

        original_place = self.remove_determinant(original_place).lower()
        move_to_place = self.remove_determinant(move_to_place).lower()

        gt_label, pred_label = None, None

        original_place_score = self.compute_lexical_overlap(prediction, original_place)
        move_to_place_score = self.compute_lexical_overlap(prediction, move_to_place)

        if original_place_score == move_to_place_score:
            pred_label = 3

        if original_place_score > move_to_place_score:
            pred_label = 1
        elif original_place_score < move_to_place_score:
            pred_label = 2

        if original_place == answer:
            gt_label = 1 
        elif move_to_place == answer:
            gt_label = 2

        return [gt_label, pred_label]

    @staticmethod
    def check_answer_for_cg_location(prediction: str, answer: str) -> list:
        prediction = prediction.lower()
        answer = answer.lower()

        if 'no' in prediction and 'yes' not in prediction:
            pred_label = 0
        elif 'yes' in prediction and 'no' not in prediction:
            pred_label = 1
        else:
            pred_label = -1

        if 'no' in answer:
            gt_label = 0 
        elif 'yes' in answer:
            gt_label = 1

        return [gt_label, pred_label]

    def check_fullness_answer(self, prediction: str, answer: str) -> list:

        prediction = prediction.replace('.', '').lower()

        less_full_answer_list = ['less full', 'emptier', 'more empty']
        more_full_answer_list = ['more full', 'fuller']

        pred_label, gt_label = None, None
        for less_full_ans in less_full_answer_list:
            if less_full_ans in prediction:
                pred_label = 1

        if not pred_label:
            for more_full_ans in more_full_answer_list:
                if more_full_ans in prediction:
                    pred_label = 2

        if not pred_label:
            if "equally full" in prediction:
                pred_label = 3

        if not pred_label:
            pred_label = -1  # corrupted

        if answer == 'less full':
            gt_label = 1 
        elif answer == 'more full':
            gt_label = 2
        elif answer == 'equally full':
            gt_label = 3

        return [gt_label, pred_label]

    def check_accessibility_answer(self, prediction: str, answer: str) -> list:

        prediction = prediction.replace('.', '').lower()

        pred_label, gt_label = None, None
        if "more accessible" in prediction:
            pred_label = 1
        elif "less accessible" in prediction:
            pred_label = 2
        elif "equally accessible" in prediction:
            pred_label = 3
        else:
            pred_label = -1  # corrupted

        if answer == 'more accessible':
            gt_label = 1 
        elif answer == 'less accessible':
            gt_label = 2
        else:
            gt_label = 3

        return [gt_label, pred_label]

    def check_attitude_answer(self, prediction: str, answer: str) -> list:

        prediction = prediction.lower()
        answer = answer.lower()

        answer_map = {
            'a': 'positive',
            'b': 'neutral',
            'c': 'negative'
        }
        prediction_token = prediction.split('\n\n')[-1].split(':')[-1].split('.')[0].strip().lower()

        gt_label, pred_label = None, None

        if answer == 'positive':
            gt_label = 1 
        elif answer == 'negative':
            gt_label = 2
        else:
            gt_label = 3

        try:
            prediction = answer_map[prediction_token]

            if prediction == 'positive':
                pred_label = 1 
            elif prediction == 'negative':
                pred_label = 2
            else:
                pred_label = 3

        except:
            if 'positive' in prediction_token and 'negative' in prediction_token:
                pred_label = -1
            elif 'positive' in prediction_token and 'neutral' in prediction_token:
                pred_label = -1
            elif 'neutral' in prediction_token and 'negative' in prediction_token:
                pred_label = -1
            elif 'positive' in prediction_token:
                pred_label = 1 
            elif 'negative' in prediction_token:
                pred_label = 2 
            elif 'neutral' in prediction_token:
                pred_label = 3 
            else:
                pred_label = -1

        return [gt_label, pred_label]


    def evaluate(self, result_path: str, location_granularity: str, perspective: str) -> dict:

        result_data = self.datautils.load_json(result_path)
        meta_data =self.datautils.load_json('../data/opentomi/opentom_v2/meta_data.json')

        # added cot prompting for entity state questions
        cot_flag, llama_flag = False, False
        if 'cot' in result_path:
            cot_flag = True
        if 'llama' in result_path:
            llama_flag = True

        # NOTE: 
        # fo -> first order; so -> second order; 0h -> 0-hop 1h -> 1-hop esq -> entity state question
        location_fo, location_so = [[] for _ in range(5)], [[] for _ in range(5)]
        multihop_fo, multihop_so = [[] for _ in range(5)], [[] for _ in range(5)]
        attitude = [[] for _ in range(5)]

        for batch_num, batch_content in result_data.items():

            cur_batch_idx = int(batch_num.split('-')[-1]) - 1

            for key, val in batch_content.items():

                mover, affected_char, eoi, original_place, move_to_place = meta_data[key]['plot_info'].values()
                places = [original_place.replace('_', ' '), move_to_place.replace('_', ' ')]

                for question_id, question_dict in val.items():

                    cur_question_type = question_dict['type']
                    question_content = question_dict['question']

                    pred_answer = question_dict['prediction'].strip()
                    gt_answer = question_dict['answer'].strip()

                    # NOTE: evaluate based on the character
                    if perspective == 'observer':
                        if mover in question_content and affected_char not in question_content:
                            continue

                        if mover in question_content and affected_char in question_content:
                            question_tokens = question_content.replace("'s", '').replace(',', '').split()

                            mover_idx = question_tokens.index(mover)
                            affected_char_idx = question_tokens.index(affected_char)

                            if mover_idx < affected_char_idx:
                                continue

                    elif perspective == 'mover':
                        if mover not in question_content and affected_char in question_content:
                            continue

                        if mover in question_content and affected_char in question_content:
                            question_tokens = question_content.replace("'s", '').replace(',', '').split()

                            mover_idx = question_tokens.index(mover)
                            affected_char_idx = question_tokens.index(affected_char)

                            if mover_idx > affected_char_idx:
                                continue

                    if cot_flag:
                        pred_answer = self.parse_cot_answer(pred_answer)

                    if cur_question_type == 'location-fo':

                        if location_granularity == 'fine':
                            gt, pred = self.check_answer_for_fg_location(pred_answer, gt_answer, original_place, move_to_place)
                        else:
                            gt, pred = self.check_answer_for_cg_location(pred_answer, gt_answer)
                        
                        location_fo[cur_batch_idx].append(tuple(('location', gt, pred)))

                    elif cur_question_type == 'location-so':
                        if location_granularity == 'fine':
                            gt, pred = self.check_answer_for_fg_location(pred_answer, gt_answer, original_place, move_to_place)
                        else:
                            gt, pred = self.check_answer_for_cg_location(pred_answer, gt_answer)

                        location_so[cur_batch_idx].append(tuple(('location', gt, pred)))

                    elif cur_question_type == 'multihop-fo':

                        if 'fullness' in question_content:
                            gt, pred = self.check_fullness_answer(pred_answer, gt_answer)

                            multihop_fo[cur_batch_idx].append(tuple(('fullness', gt, pred)))

                        elif 'accessibility' in question_content:
                            if '|' in gt_answer:
                                gt_answer = "equally accessible"

                            if isinstance(gt_answer, list):
                                gt_answer = [ele for ele in gt_answer if ele != 'corrupted']
                                assert len(gt_answer) == 1, print(key, gt_answer)
                                gt_answer = gt_answer[0]

                            gt, pred = self.check_accessibility_answer(pred_answer, gt_answer)

                            multihop_fo[cur_batch_idx].append(tuple(('accessibility', gt, pred)))

                    elif cur_question_type == 'multihop-so':
                        if 'fullness' in question_content:
                            gt, pred = self.check_fullness_answer(pred_answer, gt_answer)

                            multihop_so[cur_batch_idx].append(tuple(('fullness', gt, pred)))

                        elif 'accessibility' in question_content:

                            if '|' in gt_answer:
                                gt_answer = "equally accessible"

                            if isinstance(gt_answer, list):
                                gt_answer = [ele for ele in gt_answer if ele != 'corrupted']
                                assert len(gt_answer) == 1 
                                gt_answer = gt_answer[0]

                            gt, pred = self.check_accessibility_answer(pred_answer, gt_answer)

                            multihop_so[cur_batch_idx].append(tuple(('accessibility', gt, pred)))


                    elif cur_question_type == 'attitude':
                        gt, pred = self.check_attitude_answer(pred_answer, gt_answer)

                        attitude[cur_batch_idx].append(tuple(('attitude', gt, pred)))

        result_dict = {
            'location-fo': location_fo,
            'location-so': location_so,
            'multihop-fo': multihop_fo,
            'multihop-so': multihop_so,
            'attitude': attitude,
        }

        return result_dict
