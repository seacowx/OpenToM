import os, sys
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.utils import DataUtils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', type=str, default='../../data/extract_plot/tomi_chatgpt_3_shot_6991844.json'
    )
    return parser.parse_args()


def main():
    args = get_args()
    datautils = DataUtils()

    n_shot = int(args.data_path.split('_')[3])
    data = datautils.load_json(args.data_path)

    for i in range(n_shot + 1, len(data) + 1):
        cur_plot = data[str(i)]['plot']
        cur_extracted_plot = data[str(i)]['extracted_plot']
        cur_narrative = data[str(i)]['gpt_narrative']


        print(cur_narrative)
        print(' ')
        print(cur_plot)
        print(' ')
        print(cur_extracted_plot)
        print('\n\n')



if __name__ == '__main__':
    main()
