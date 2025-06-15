import json
import argparse
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from eval_scripts.utils import extract_boxed_1, replace_outermost_boxed
res_list = []
def main(args):
    src1 = f'train_datasets/{args.data_name}/{args.data_name}_distill.json'  
    src2 = f'train_datasets/{args.data_name}/{args.data_name}_distill_output.json'  
    res_path = f'train_datasets/{args.data_name}/{args.data_name}_distill_final.json'
    with open(src1, "r") as f:
        src1_data = json.load(f)
    with open(src2, "r") as f:
        src2_data = json.load(f)
    correct = 0
    remove_count = 0
    wrong_list = []
    for i in range(len(src2_data)):
        ground_truth = src2_data[i]['output']
        prediction = extract_boxed_1(src2_data[i]["predict"])
        if prediction == []:
            remove_count += 1
            continue
        
        # print(f"Debug: Prediction: '{prediction}', Ground Truth: '{ground_truth}'")  # Debug output
        if prediction[0].lower() == ground_truth.lower():
            correct += 1
            new_item = src1_data[i].copy()
            new_item['thinking'] = None
            new_item['solution'] = src2_data[i]["predict"]
            res_list.append(new_item)
        else:
            new_item = src1_data[i].copy()
            res_list.append(new_item)              
    for item in wrong_list:
        print(f"pred:{item['pred']}, tgt:{item['answer']}")
    print("correct:", correct)
    print('remove:', remove_count)        
    print("accuracy:", correct/(len(src2_data)-remove_count))
    print(len(res_list))
    with open(res_path, 'w', encoding='utf-8') as fw:
        json.dump(res_list, fw, indent=2)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_name', type=str, default='hitab', help='')
    parser.add_argument('--model_name', type=str, default='deepseek', help='')    
    args = parser.parse_args()
    args.data_name = 'tabfact'
    main(args)   
