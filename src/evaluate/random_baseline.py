import os, sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from glob import glob
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

from utils.utils import DataUtils


def main():
    datautils = DataUtils()
    labelencoder = LabelEncoder()

    valid_q_types = (
        'location_cg_fo', 
        'location_fg_fo', 
        'location_cg_so', 
        'location_fg_so', 
        'multihop_fo', 
        'multihop_so',
        'attitude'
    )

    # set seed to random baseline
    np.random.seed(42)

    all_questions = {}
    for q_type in valid_q_types:
        new_questions = datautils.load_json(os.path.join('../../data/opentomi/opentom_v2/', f'{q_type}.json'))

        cur_answers = []
        for key_val in new_questions.values():
            for q_val in key_val:
                cur_answers.append(q_val['answer'])

        all_questions[q_type] = cur_answers

    for q_type, q_answers in all_questions.items():
        
        if 'multihop' in q_type:

            q_answers1 = [ele for ele in q_answers if 'full' in ele]
            q_answers2 = [ele for ele in q_answers if 'full' not in ele]

            q_answers1 = labelencoder.fit_transform(q_answers1)
            total_labels = len(labelencoder.classes_)

            random_prediction = []
            for _ in range(len(q_answers1)):
                random_prediction.append(np.random.randint(total_labels))

            acc1 = accuracy_score(q_answers1, random_prediction)
            f11 = f1_score(q_answers1, random_prediction, average="macro")

            print('===================================================')
            print(f'Question Type: {q_type}_fullness')
            print('===================================================')
            print(f'Accuracy: {acc1:.3f}')
            print(f'F1 Score: {f11:.3f}')
            print('===================================================')
            print('\n')

            q_answers2 = labelencoder.fit_transform(q_answers2)
            total_labels = len(labelencoder.classes_)

            random_prediction = []
            for _ in range(len(q_answers2)):
                random_prediction.append(np.random.randint(total_labels))

            acc2 = accuracy_score(q_answers2, random_prediction)
            f12 = f1_score(q_answers2, random_prediction, average="macro")

            print('===================================================')
            print(f'Question Type: {q_type}_accessibility')
            print('===================================================')
            print(f'Accuracy: {acc2:.3f}')
            print(f'F1 Score: {f12:.3f}')
            print('===================================================')
            print('\n')

            print('===================================================')
            print(f'Question Type: {q_type}_overall')
            print('===================================================')
            print(f'Accuracy: {np.mean([acc1, acc2]):.3f}')
            print(f'F1 Score: {np.mean([f11, f12]):.3f}')
            print('===================================================')
            print('\n')


        else:

            q_answers = labelencoder.fit_transform(q_answers)
            total_labels = len(labelencoder.classes_)

            random_prediction = []
            for _ in range(len(q_answers)):
                random_prediction.append(np.random.randint(total_labels))

            print('===================================================')
            print(f'Question Type: {q_type}')
            print('===================================================')
            print(f'Accuracy: {accuracy_score(q_answers, random_prediction):.3f}')
            print(f'F1 Score: {f1_score(q_answers, random_prediction, average="macro"):.3f}')
            print('===================================================')
            print('\n')


if __name__ == '__main__':
    main()
