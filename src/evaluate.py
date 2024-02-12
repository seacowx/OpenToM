import sys
import math
import argparse
import numpy as np
from numpy.lib import average

sys.path.append('./evaluate/')
from opentom_evaluator import OpenToMEvaluator

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result_path', type=str, required=True, help='path to the result file',
    )
    parser.add_argument(
        '-lg', '--location_granularity', type=str, default='fine', help='fine or coarse',
    )
    parser.add_argument(
        '--perspective', type=str, default='all', help='chosee between "all", "mover", and "observer"'
    )
    return parser.parse_args()


def main():
    args = get_args()
    opentom_evaluator = OpenToMEvaluator()
    result_dict = opentom_evaluator.evaluate(args.result_path, args.location_granularity, args.perspective)

    question_types = [
        'location-fo',
        'location-so',
        'multihop-fo',
        'multihop-so',
        'attitude'
    ]

    for question_type in question_types:
        result_subdict = result_dict[question_type]

        if not result_subdict[0]:
            continue

        acc_list, f1_list, corrupt_count_list = [], [], []
        acc_list2, f1_list2, corrupt_count_list2 = [], [], []
        acc_list3, f1_list3 = [], []

        for batch_result in result_subdict:

            pred_list, gt_list = [], []
            pred_list2, gt_list2 = [], []
            
            for entry in batch_result:

                cur_type = entry[0]

                if cur_type == 'fullness':
                    gt_list.append(entry[1])
                    pred_list.append(entry[2])
                elif cur_type == 'accessibility':
                    gt_list2.append(entry[1])
                    pred_list2.append(entry[2])
                else:
                    gt_list.append(entry[1])
                    pred_list.append(entry[2])

            if pred_list2:
                valid_pred = [ele for ele in pred_list if ele != -1]
                valid_gt = [gt_list[i] for i in range(len(pred_list)) if pred_list[i] != -1]
                valid_pred = [valid_pred[i] for i in range(len(valid_gt)) if valid_gt[i] != None]
                valid_gt = [ele for ele in valid_gt if ele != None]

                pred_corrupted = (len(pred_list) - len(valid_pred)) / len(pred_list)
                corrupt_count_list.append(pred_corrupted)

                valid_pred2 = [ele for ele in pred_list2 if ele != -1]
                valid_gt2 = [gt_list2[i] for i in range(len(pred_list2)) if pred_list2[i] != -1]
                valid_pred2 = [valid_pred2[i] for i in range(len(valid_gt2)) if valid_gt2[i] != None]
                valid_gt2 = [ele for ele in valid_gt2 if ele != None]
                pred_corrupted = (len(pred_list2) - len(valid_pred2)) / len(pred_list2)
                corrupt_count_list2.append(pred_corrupted)
            else:
                valid_pred = [ele for ele in pred_list if ele != -1]
                valid_gt = [gt_list[i] for i in range(len(pred_list)) if pred_list[i] != -1]
                valid_pred = [valid_pred[i] for i in range(len(valid_gt)) if valid_gt[i] != None]
                valid_gt = [ele for ele in valid_gt if ele != None]

                pred_corrupted = (len(pred_list) - len(valid_pred)) / len(pred_list)
                corrupt_count_list.append(pred_corrupted)

            assert len(valid_pred) == len(valid_gt)

            if pred_list2:
                acc = accuracy_score(valid_gt, valid_pred)
                f1 = f1_score(valid_gt, valid_pred, average='macro')
                acc2 = accuracy_score(valid_gt2, valid_pred2)
                f12 = f1_score(valid_gt2, valid_pred2, average='macro')

                acc3 = accuracy_score(valid_gt + valid_gt2, valid_pred + valid_pred2)
                f13 = f1_score(valid_gt + valid_gt2, valid_pred + valid_pred2, average='macro')

                acc_list.append(acc)
                f1_list.append(f1)

                acc_list2.append(acc2)
                f1_list2.append(f12)

                acc_list3.append(acc3)
                f1_list3.append(f13)

            else:
                acc = accuracy_score(valid_gt, valid_pred)
                f1 = f1_score(valid_gt, valid_pred, average='macro')

                acc_list.append(acc)
                f1_list.append(f1)

        if acc_list2:
            avg_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)

            avg_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list)

            avg_corrupt = np.mean(corrupt_count_list)

            print('============================================')
            print(f'Question type: {question_type}_fullness')
            print(f'Corrupted generation: {avg_corrupt * 100}%')
            print(f'Avearge Accuracy: {avg_acc:.3f}, Variance: {std_acc:.3f}')
            print(f'Average F1: {avg_f1:.3f}, Variance: {std_f1:.3f}')
            print('============================================')
            print('\n')

            avg_acc = np.mean(acc_list2)
            std_acc = np.std(acc_list2)

            avg_f1 = np.mean(f1_list2)
            std_f1 = np.std(f1_list2)

            avg_corrupt = np.mean(corrupt_count_list2)

            print('============================================')
            print(f'Question type: {question_type}_accessibility')
            print(f'Corrupted generation: {avg_corrupt * 100}%')
            print(f'Avearge Accuracy: {avg_acc:.3f}, Variance: {std_acc:.3f}')
            print(f'Average F1: {avg_f1:.3f}, Variance: {std_f1:.3f}')
            print('============================================')
            print('\n')

            avg_acc = np.mean(acc_list3)
            std_acc = np.std(acc_list3)

            avg_f1 = np.mean(f1_list3)
            std_f1 = np.std(f1_list3)

            avg_corrupt = np.mean(corrupt_count_list + corrupt_count_list2)

            print('============================================')
            print(f'Question type: {question_type}_overall')
            print(f'Corrupted generation: {avg_corrupt * 100}%')
            print(f'Avearge Accuracy: {avg_acc:.3f}, Variance: {std_acc:.3f}')
            print(f'Average F1: {avg_f1:.3f}, Variance: {std_f1:.3f}')
            print('============================================')
            print('\n')

        else:
            avg_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)

            avg_f1 = np.mean(f1_list)
            std_f1 = np.std(f1_list)

            avg_corrupt = np.mean(corrupt_count_list)

            print('============================================')
            print(f'Question type: {question_type}')
            print(f'Corrupted generation: {avg_corrupt * 100}%')
            print(f'Avearge Accuracy: {avg_acc:.3f}, Variance: {std_acc:.3f}')
            print(f'Average F1: {avg_f1:.3f}, Variance: {std_f1:.3f}')
            print('============================================')
            print('\n')


if __name__ == "__main__":
    main()
