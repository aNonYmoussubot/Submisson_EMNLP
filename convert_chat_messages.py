# src_path = '/root/autodl-tmp/datasets/Table-Reasoning-data/table_reasoning_data.json'
src_path = '/xxx/table_reasoning_data.json'
res_path = '/xxx/table_reasoning_data1.json'
import json
from tqdm import tqdm
with open(src_path, 'r', encoding='utf-8') as fr:
    table_content = json.load(fr)
import random
res_list = []

for idx, item in tqdm(enumerate(table_content)):
    instruction_seg = item['instruction']
    table_seg = item['input_seg']
    question = item['question']
    thinking = item['thinking']
    solution = item['solution']
    prompt_user = f'''Instruction": "{instruction_seg}\n"table": "{table_seg}"\n "question": "{question}"\n"'''
    if thinking == None:
        prompt_assistant = f'''<think>\n\n</think>\n{solution}'''
    else:
        prompt_assistant = f'''<think>\n{thinking}</think>\n{solution}'''
    messages = [
       
        {"role": "user", "content": prompt_user},
        {"role": "assistant", "content": prompt_assistant},
        
    ]
    new_item = item.copy()
    new_item['messages'] = messages
    res_list.append(new_item)

random.shuffle(res_list)
print(res_list[0]['messages'])
with open(res_path, 'w', encoding='utf-8') as fw:
    json.dump(res_list, fw, indent=2)
