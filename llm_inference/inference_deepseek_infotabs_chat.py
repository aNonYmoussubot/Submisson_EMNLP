import os
import json
import sys
from openai import OpenAI
from tqdm import tqdm
import argparse
DEEPSEEK_API_KEY = 'xxx'
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def main(args):
    output_folder = f'output/test/{args.model_name}' 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_path = f'test_datasets/{args.data_name}/test_1k.json'
    # loading ....
    with open(file_path, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)


    for idx, item in tqdm(enumerate(data_loaded),total=len(data_loaded)):
        instruction_seg = item['instruction']
        table_seg = item['input_seg']
        question_seg = item['question']
        
        template = f'''
        You will be given a question. Please reason step by step, and put your final answer within \\boxed{{}}
        "Instruction": "{instruction_seg}\n"table": "{table_seg}"\n "question": "{question_seg}"\n"'''
        
        try:
            completion = client.chat.completions.create(
                model="deepseek-chat", #deepseek-reasoner
                messages=[
                    {"role": "system", "content": "You are an expert in table question answer!"},
                    {"role": "user", "content": template},
                ],
            )
        except Exception as e:
            print(f'Error processing {idx}: {e}')
            continue

        res = completion.choices[0].message.content
        print(res)
        item['predict'] = res

    output_file_path = f"output/test/{args.model_name}/{args.data_name}_test_1k.json"
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(data_loaded, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_name', type=str, default='hitab', help='')
    parser.add_argument('--model_name', type=str, default='deepseek', help='')    
    args = parser.parse_args()
    args.data_name = 'infotabs'
    args.model_name = 'deepseek-r1'      
    main(args)   