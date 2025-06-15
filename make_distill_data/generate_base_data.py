import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import re
import json
import sys
import math
import torch
import argparse
import transformers
from transformers import GenerationConfig
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, PeftModel
from typing import Dict, Optional, Sequence
from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from trl import SFTTrainer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    pipeline
)
from datasets import load_dataset
device_map = {"": 0}

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--base_model', type=str, default="/data1/pretrained-models/llama-7b-hf")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--context_size', type=int, default=-1, help='context size during fine-tuning')
    parser.add_argument('--flash_attn', type=bool, default=False, help='')
    parser.add_argument('--temperature', type=float, default=0.6, help='')
    parser.add_argument('--top_p', type=float, default=0.9, help='')
    parser.add_argument('--max_gen_len', type=int, default=512, help='')
    parser.add_argument('--input_data_file', type=str, default='input_data/', help='')
    parser.add_argument('--output_data_file', type=str, default='output_data/', help='')
    parser.add_argument('--trained_model', type=str, default='')
    parser.add_argument('--datasets', type=str, default='')   
    parser.add_argument('--peft_flag', type=bool, default=False) 
    args = parser.parse_args()
    return args

def build_generator(
    item, model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
):
    def response(item):
        prompt = item 
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            use_cache=use_cache
        )
        out = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        out = out.split(prompt)[1].strip()
        return out
    return response

def main(args):
    def create_message_column(row):
        messages = []
        user = {
            "content": f"Instruction: {row['instruction']}\ntable: {row['input_seg']}\n question: {row['question']} Please reason step by step, and put your final answer within \\boxed{{}}\n",
            "role": "user"
        }        
        messages.append(user)        
        assistant = {
            "content": f"{row['output']}",
            "role": "assistant"
        }        
        messages.append(assistant)
        return {"messages": messages}
    def format_dataset_chatml(row):
        return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=False)}

    def test_inference(prompts, batch_size=4):
        formatted_prompts = [
            pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
            for prompt in prompts
        ]
        
        outputs = pipe(
            formatted_prompts,
            max_new_tokens=8192,
            do_sample=True,
            num_beams=1,
            temperature=0.4,
            batch_size=batch_size, 
        )
        
        results = []
        for prompt, output in zip(formatted_prompts, outputs):
            generated_text = output[0]['generated_text'][len(prompt):].strip()
            results.append(generated_text)
        
        return results

    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    base_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        model_max_length=args.context_size,
        padding_side="left",
        add_eos_token=True,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = base_model


    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    dataset = load_dataset('json', data_files=args.input_data_file, split = f'train')
    dataset_chatml = dataset.map(create_message_column)
    dataset_chatml = dataset_chatml.map(format_dataset_chatml)
    print(dataset_chatml[0])
    batch_size = 8  
    test_data_pred = []
    for i in tqdm(range(0, len(dataset_chatml), batch_size)):
        batch = dataset_chatml[i:i + batch_size]
        prompts = [item[0]['content'] for item in batch['messages']]
        outputs = test_inference(prompts, batch_size=batch_size)

        for j, output in enumerate(outputs):
            new_item = {}
            answer_content = output                      
            new_item["idx"] = i + j
            new_item["output"] = batch['output'][j]
            new_item["predict"] = answer_content
            test_data_pred.append(new_item)      
      
    
    
    with open(args.output_data_file, "w") as f:
        json.dump(test_data_pred, f, indent = 2)


if __name__ == "__main__":
    args = parse_config()
    args.base_model = "/xxx/Qwen2.5-14B-Instruct"
    args.datasets = 'hitab'# 'wikitq', 'tabfact', 'infotabs'
    args.max_gen_len = 8196
    args.input_data_file = f"train_datasets/{args.datasets}/{args.datasets}_distill.json"
    args.output_data_file = f"train_datasets/{args.datasets}/{args.datasets}_output.json"
    main(args)

