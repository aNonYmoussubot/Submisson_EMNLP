import json
from tqdm import tqdm
import re
import unicodedata
import math
import sys
import os
import argparse
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from eval_scripts.utils import llm_check, extract_boxed_1, replace_outermost_boxed
def hmt_score(prediction, answer):
    prediction = hmt_process_answer(prediction)
    answer = hmt_process_answer(answer)
    if hmt_equal(prediction, answer):
        return 1.0
    else:
        return 0.0


def hmt_equal(prediction, answer):
    if answer == '$40,000':
        pass
    # import pdb
    # pdb.set_trace()
    try:
        if prediction.split(",")[0] not in ['inf', 'infinity', 'INF', 'INFINITY', 'True', 'NAN', 'nan', 'False', '-inf', '-INF', '-INFINITY', '-infinity', 'NaN', 'Nan'] and float(prediction.split(",")[0]):
            prediction = float(prediction.split(",")[0])
    except:
        prediction = prediction

    if type(prediction) != type(answer):
        return False
    if isinstance(prediction, str):
        # print("str:", answer, prediction)
        return prediction.find(answer) == 0
    if isinstance(prediction, int) or isinstance(prediction, float):
        # return math.fabs(prediction - answer) < 1e-5
        # print("float/int:", answer, prediction)
        return math.fabs(prediction - answer) < 1e-5
    if isinstance(prediction, list):
        if len(prediction) != len(answer):
            return False
        return all([hmt_equal(prediction[i], answer[i]) for i in range(len(prediction))])



def naive_str_to_float(string):
    """ A naive way to convert str to float, if convertable."""
    sanitized = string
    try:
        if sanitized[0] == '(':
            sanitized = sanitized[1:]
        if (sanitized[-1] == '%') or (sanitized[-1] == ')'):
            sanitized = sanitized[: -1]
        sanitized = sanitized.replace(',', '').replace('$', '')

        new = float(sanitized)
        return new
    except:
        return normalize(string)


def normalize(x):
    """ Normalize header string. """
    # Copied from WikiTableQuestions dataset official evaluator.
    if x is None:
        return None
    # Remove diacritics
    x = x.replace('$','')
    x = ''.join(c for c in unicodedata.normalize('NFKD', x)
                if unicodedata.category(c) != 'Mn')
    # Normalize quotes and dashes
    x = re.sub("[‘’´`]", "'", x)
    x = re.sub("[“”]", "\"", x)
    x = re.sub("[‐‑‒–—−]", "-", x)
    while True:
        old_x = x
        # Remove citations
        x = re.sub("((?<!^)\[[^\]]*\]|\[\d+\]|[•♦†‡*#+])*$", "", x.strip())
        # Remove details in parenthesis
        x = re.sub("(?<!^)( \([^)]*\))*$", "", x.strip())
        # Remove outermost quotation mark
        x = re.sub('^"([^"]*)"$', r'\1', x.strip())
        if x == old_x:
            break
    # Remove final '.'
    if x and x[-1] == '.':
        x = x[:-1]
    # Collapse whitespaces and convert to lower case
    x = re.sub('\s+', ' ', x, flags=re.U).lower().strip()
    #x = x.replace('|', ', ')
    x = x.replace('\\text{ ', '').replace('\\text{', '')#.replace('\\%', '')
    # x = x.replace('\\','')
    # x = x.replace('\\','')    

    return x

def hmt_score(prediction, answer):
    prediction = hmt_process_answer(prediction)
    answer = hmt_process_answer(answer)
    if hmt_equal(prediction, answer):
        return 1.0
    else:
        return 0.0
    
def hmt_process_answer(answer):
    """ 4 types of answer: 1)region; 2)num_list(aggr); 3)header_list(argmax); 4)num(count/div)"""
    if isinstance(answer, int) or isinstance(answer, float):
        return float(answer)
    if isinstance(answer, str):
        return naive_str_to_float(answer.strip().lower())
    if isinstance(answer, list):
        if isinstance(answer[0], list):  # pred region
            if len(answer) == 1 and len(answer[0]) == 1:  # pred region with one cell, flatten
                return hmt_process_answer(answer[0][0])
            elif len(answer) == 1:  # pred region with one line
                return hmt_process_answer(answer[0])
            elif len(answer[0]) == 1:  # pred region with one line
                return hmt_process_answer([row[0] for row in answer])
            else:  # pred region is a matrix
                return [hmt_process_answer(a) for a in answer]
        else:  # list or processed single-line region
            if len(answer) == 1:  # answer with one cell or pred list
                return hmt_process_answer(answer[0])
            else:
                return [hmt_process_answer(a) for a in answer]

def main(args):
    src1 = f'train_datasets/{args.data_name}/{args.data_name}_distill.json'  
    src2 = f'train_datasets/{args.data_name}/{args.data_name}_distill_output.json'  
    res_path = f'train_datasets/{args.data_name}/{args.data_name}_distill_final.json'
    with open(src1, "r") as f:
        src1_data = json.load(f)
    with open(src2, "r") as f:
        src2_data = json.load(f)    
    res_list = []
    wrong_list = []
    corrects = 0
    remove_count = 0
    pred_list = []
    for idx, each_line in tqdm(enumerate(zip(src1_data, src2_data)), total=len(src1_data)):
        src1 = each_line[0]
        src2 = each_line[1]
        question = src1['question']
        target = src1['output']
        solution = src2['predict']
        predict_list= extract_boxed_1(solution)
        if predict_list != []:
            predict = predict_list[0]
            # base model inference success!
            if hmt_score(predict.lower(), target) == 1:
                corrects+=1
                new_item = src1.copy()
                new_item['thinking'] = None
                new_item['solution'] = solution
                res_list.append(new_item)
            else:
                print(f"error predict:{predict.lower()}, target:{target}")
                res = llm_check(predict.lower(), target, question)
                print(res)
                if res == 'True' or res =='true':
                    updated_text = replace_outermost_boxed(solution, target)
                    corrects+= 1
                    new_item = src1.copy()
                    new_item['thinking'] = None
                    new_item['solution'] = updated_text
                    res_list.append(new_item)                    
                    pred_list.append(1)
                else:
                    print(f"error predict:{predict.lower()}, target:{target}")
                    pred_list.append(0)
                    new_item = src1.copy()
                    res_list.append(new_item)                    
        else:
            remove_count += 1
            print("No match found.")
    correct_percent = corrects / (len(src1_data)-remove_count)
    print(correct_percent)
    with open(res_path, 'w', encoding='utf-8') as fw:
        json.dump(res_list, fw, indent=2)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_name', type=str, default='hitab', help='')
    parser.add_argument('--model_name', type=str, default='deepseek', help='')    
    args = parser.parse_args()
    args.data_name = 'wikitq' 
    main(args)        