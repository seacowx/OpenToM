import json
from copy import deepcopy


metadata = json.load(open('./opentom_data/meta_data_long.json'))

sub_folders = [
    'attitude', 
    'multihop_fo', 
    'multihop_so',
    'location_cg_fo',
    'location_cg_so',
    'location_fg_fo',
    'location_fg_so',
]


new_data = []
question_types = []
for task_type in sub_folders:

    task_data = json.load(open(f'../data/opentom_data/{task_type}.json'))

    for key, entry in task_data.items():

        if key in metadata.keys():
            cur_meta_data = metadata[key]

            if 'long_narrative' not in cur_meta_data.keys():
                print(1)

            for q_entry in entry:

                new_val = deepcopy(cur_meta_data)

                new_val['question'] = q_entry

                if q_entry['type'] not in question_types:
                    question_types.append(q_entry['type'])

                new_data.append(new_val)

print(question_types)
print(len(new_data))

# json.dump(new_data, open('../data/opentom_long.json', 'w'), indent=4)
