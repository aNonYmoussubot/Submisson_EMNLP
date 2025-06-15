import json
import os
from openai import OpenAI
from tqdm import tqdm
ARK_API_KEY='xxx'
client = OpenAI(api_key=ARK_API_KEY, base_url="https://api.deepseek.com")
input_folder = 'xxx'
output_folder = 'xxx'
json_files = ['wikitq.json', 'tabfact.json', 'infotabs.json', 'hitab.json']
for json_file in json_files:
    file_path = os.path.join(input_folder, json_file)             
    new_data_list = []
    with open(file_path, 'r') as f:
        data_loaded = json.load(f)

    for idx, item in tqdm(enumerate(data_loaded), total=len(data_loaded)):
        current_item = item.copy()
        instruction_seg = item['instruction']
        table_seg = item['input_seg']
        question_seg = item['question']
        template = f"""You will be given a question. Please reason step by step, and put your final answer within \\boxed{{}}:
"Instruction": "{instruction_seg}",
"Table": "{table_seg}",
"Question": "{question_seg}"
"""
        try:
            completion = client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert in table question answer!"},
                    {"role": "user", "content": template},
                ],
                max_tokens=8192,
                top_p=0.6,
            )
        except Exception as e:
            print(f'Error processing {json_file}: {e}')
            continue

        res = completion.choices[0].message.content
        thinks = completion.choices[0].message.reasoning_content
        print(res)
        current_item['solution'] = res
        current_item['thinking'] = thinks
        new_data_list.append(current_item)

    output_file_path = os.path.join(output_folder, json_file)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(new_data_list, file, indent=4)