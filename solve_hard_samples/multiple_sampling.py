import os
import json
import sys
from openai import OpenAI
from tqdm import tqdm
import argparse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from eval_scripts.utils import extract_boxed_1, eval_one
DEEPSEEK_API_KEY = 'xxx'
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def main(args):
    file_path = f'train_datasets/{args.data_name}/{args.data_name}_hard_small.json' 

    with open(file_path, 'r', encoding='utf-8') as f:
        data_loaded = json.load(f)
    solved_data = []
    for idx, item in tqdm(enumerate(data_loaded),total=len(data_loaded)):
        solved_flag = False
        instruction_seg = item['instruction']
        table_seg = item['input_seg']
        question_seg = item['question']
        answer = item['output']
        
        template = f'''
        You will be given a question. Please reason step by step, and put your final answer within \\boxed{{}}
        "Instruction": "{instruction_seg}\n"table": "{table_seg}"\n "question": "{question_seg}"\n"'''
        for temp in args.temperature:
            try:
                completion = client.chat.completions.create(
                    model="deepseek-reasoner", 
                    messages=[
                        {"role": "system", "content": "You are an expert in table question answer!"},
                        {"role": "user", "content": template},
                    ],
                    max_tokens=8192,
                    temperature=temp,
                )
            except Exception as e:
                print(f'Error processing {idx}: {e}')
                continue
            solution = completion.choices[0].message.content
            thinks = completion.choices[0].message.reasoning_content
            res = eval_one(args.data_name, solution, answer)
            if res == True:
                item['solution'] = solution
                item['thinking'] = thinks
                solved_data.append(item)
                print('current sample is solved!')
                solved_flag = True
                break
        if solved_flag == False:
            print('current sample is not solved!')

    output_file_path = f'train_datasets/{args.data_name}/{args.data_name}_hard_small_solved.json' 
    with open(output_file_path, 'w', encoding='utf-8') as file:
        json.dump(solved_data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_name', type=str, default='hitab', help='')
    parser.add_argument('--temperature', type=list, default=[0.6], help='')    
    args = parser.parse_args()
    args.data_name = 'tabfact'
    args.temperature = [1.0, 1.4, 1.5, 1.6, 1.7, 1.8]    
    main(args)   