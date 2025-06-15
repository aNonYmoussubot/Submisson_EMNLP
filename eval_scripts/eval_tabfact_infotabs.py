import json
import argparse
from utils import extract_boxed

def main(args):
    src1 = f'output/test/{args.model_name}/{args.data_name}_test_1k.json'    
    with open(src1, "r") as f:
        data = json.load(f)

    correct = 0
    remove_count = 0
    wrong_list = []
    for i in range(len(data)):
        ground_truth = data[i]['answer']
        prediction = extract_boxed(data[i]["predict"])
        if prediction == None:
            remove_count += 1
            continue
        
        if prediction.lower() == ground_truth.lower():
            correct += 1
        else:
            wrong_item = {}
            wrong_item["idx"] = i
            wrong_item["answer"] = ground_truth
            wrong_item["pred"] = prediction
            wrong_list.append(wrong_item)

    print("correct:", correct)
    print('remove:', remove_count)        
    print("accuracy:", correct/(len(data)-remove_count))
    for item in wrong_list:
        print(f"pred:{item['pred']}, tgt:{item['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_name', type=str, default='hitab', help='')
    parser.add_argument('--model_name', type=str, default='deepseek', help='')    
    args = parser.parse_args()
    args.data_name = 'infotabs'
    args.model_name = 'deepseek-r1'    
    main(args)   
