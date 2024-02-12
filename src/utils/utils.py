import re
import os
import json
import yaml
import pickle


class DataUtils():

    @staticmethod
    def check_file_existence(fpath):
        '''
        check_file_existence function to check if file exists

        Args:
            fpath (str): path to the file

        Returns:
            bool: True if file exists, False otherwise
        '''
        if os.path.exists(fpath):
            raise Exception('File already tagged')

    @staticmethod
    def save_json(file: dict, fpath: str) -> None:
        '''
        save_json function to save dictionary as json file

        Args:
            file (dict): dictionary to be saved
            fpath (str): path to the json file
        '''
        with open(fpath, 'w') as f:
            json.dump(file, f, indent=4)
        f.close()

    @staticmethod
    def load_tomi(fpath: str) -> dict:
        '''
        load_txt function to load local txt file

        Args:
            fpath (str): path to the txt file

        Returns:
            dict: txt file as a dictionary
        '''
        with open(fpath, 'r') as f:
            raw_data = f.readlines()
        
        data = {}
        counter = 0
        for entry in raw_data:
            if entry.strip().split()[0] == '1':
                if counter != 0:
                    temp_entry = [e.strip() for e in temp_entry]
                    temp_content = '\n'.join(temp_entry[:-1])
                    temp_question, temp_answer = temp_entry[-1].split('?')
                    data[str(counter)] = {
                        'content': temp_content,
                        'question': temp_question.strip() + '?',
                        'answer': temp_answer.strip()
                    }
                counter += 1
                temp_entry = [entry]
            else:
                temp_entry.append(entry.strip())

        return data

    @staticmethod
    def load_txt(fpath: str) -> str:
        '''
        load_txt function to load local txt file

        Args:
            fpath (str): path to the txt file

        Returns:
            str: txt file as a string
        '''
        with open(fpath, 'r') as f:
            data = f.read()
        f.close()

        return data

    @staticmethod
    def load_json(fpath: str) -> dict:
        '''
        load_json function to load json file

        Args:
            fpath (str): path to the json file

        Returns:
            dict: json file as a dictionary
        '''
        with open(fpath, 'r') as f:
            data = json.load(f)
        f.close()

        return data

    @staticmethod
    def load_jsonl(fpath: str) -> list:
        '''
        load_jsonl function to load jsonl file

        Args:
            fpath (str): path to the jsonl file

        Returns:
            list: jsonl file loaded as list of dictionaries
        '''

        new_data = []

        with open(fpath, 'r') as f:
            raw_data = f.readlines()
            for line in raw_data:
                cur_line = json.loads(line)
                new_data.append(cur_line)

        return new_data
    
    @staticmethod
    def load_yaml(fpath: str) -> dict:
        '''
        load_yaml function to load yaml file

        Args:
            fpath (str): path to the yaml file

        Returns:
            dict: yaml file as a dictionary
        '''
        with open(fpath, 'r') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        f.close()

        return data

    @staticmethod 
    def save_pickle(file: list, path: str) -> None:
        '''
        save_pickle function to save list as pickle File

        Args:
            file (list): list to be saved
            path (str): path to the pickle File

        Returns:
            None
        '''
        with open(path, 'wb') as f:
            pickle.dump(file, f)
        f.close()

    @staticmethod
    def load_pickle(path: str) -> list:
        '''
        load_pickle function to load pickle File

        Args:
            path (str): path to the pickle File

        Returns:
            list: pickle File as a list
        '''
        with open(path, 'rb') as f:
            data = pickle.load(f)
        f.close()

        return data


class TomiUtils():

    @staticmethod
    def question_to_narrative(question: str) -> str:
        if 'really' in question:
            matched = re.match(r'Where is the ([a-z]*) really?', question)
            eoi = matched.group(1)
            new_narrative = f"At the end of the story, the {eoi} is located at "


class BaselineLabels():

    @property
    def fullness_labels(self) -> list[str]:
        return ['less full', 'equally full', 'more full']

    @property
    def weight_labels(self) -> list[str]:
        return ['lighter', 'equally heavy', 'heavier']

    @property
    def accessibility_labels(self) -> list[str]:
        return ['directly accessible', 'sealed in a container']
