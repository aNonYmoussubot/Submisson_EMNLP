import re
import math
import numpy as np
import re
from openai import OpenAI
from tqdm import tqdm
ARK_API_KEY='xxx'

def llm_check(predict, target, question=None):
    template = f'''Please determine whether the prediction and the target result are semantically consistent. If they are, return \boxed{True}; otherwise, return \boxed{False}.
Question:{question}, Prediction:{predict}, Target:{target}'''
    client = OpenAI(api_key=ARK_API_KEY, base_url="https://api.deepseek.com")
    completion = client.chat.completions.create(
            model="deepseek-chat", #"deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert in table question answer!"},
                {"role": "user", "content": template},
            ],
        )

    # 获取API的响应
    res = completion.choices[0].message.content
    match = re.search(r'\\boxed\{(.*?)\}', res)
    if match:
        predict = match.group(1)
    return predict


def llm_check_fuzzy(predict, target):
    template = f'''The motivation is that the predicted results may contain noise, leading to semantic consistency with the target results but deviations in form. Such semantically consistent predictions should still be considered correct. Please determine whether the prediction and the target result are semantically consistent. If they are, return \boxed{True}; otherwise, return \boxed{False}.
Prediction:{predict},Target:{target}'''
    client = OpenAI(api_key=ARK_API_KEY, base_url="https://api.deepseek.com")
    completion = client.chat.completions.create(
            model="deepseek-chat", 
            messages=[
                {"role": "system", "content": "You are an expert in table question answer!"},
                {"role": "user", "content": template},
            ],
        )

    res = completion.choices[0].message.content
    match = re.search(r'\\boxed\{(.*?)\}', res)
    if match:
        predict = match.group(1)
    return predict


def extract_boxed(text):
    start_token = "\\boxed{"
    end_token = "}"

    start_idx = text.find(start_token)
    if start_idx == -1:
        return None

    start_idx += len(start_token)
    end_idx = text.find(end_token, start_idx)

    if end_idx == -1:
        return None

    return text[start_idx:end_idx]

def extract_boxed_1(text: str):

    results = []
    i = 0
    marker = r'\boxed{'
    mlen   = len(marker)

    while i < len(text):
        if text.startswith(marker, i):
            # 进入 boxed 解析状态
            j      = i + mlen          # j 指向第一个 '{' 之后的位置
            depth  = 1                 # 已读到一个 '{'
            start  = j                 # 内部内容起点
            while j < len(text) and depth:
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                j += 1
            if depth == 0:             # 成功配对
                results.append(text[start:j-1])  # j 已指向匹配 '}' 之后
                i = j                  # 继续向后搜索
            else:                      # 括号不匹配，提前结束
                break
        else:
            i += 1
    return results

def replace_outermost_boxed(text, new_content):
    start = text.find(r'\boxed{')
    if start == -1:
        return text  # 没有boxed结构

    i = start + len(r'\boxed{')
    brace_count = 1
    while i < len(text):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end = i
                break
        i += 1
    else:
        raise ValueError("Unmatched braces in \\boxed{} structure")

    # 构造新的文本
    return text[:start] + f'\\boxed{{{new_content}}}' + text[end+1:]

def eval_one(data_name, predict, answer, question=None):
    pred = extract_boxed_1(predict)
    if pred == []:
        return False            
    else:  
        pred = pred[0]    
        res = llm_check(pred, answer, question)
        if res == 'True' or res =='true':
            return True
        else:
            print(f'pred:{pred}, tgt:{answer}')
            return False

        
    