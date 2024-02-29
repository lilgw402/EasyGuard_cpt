import os
import json
import pandas as pd
def process_cot_result(folder):
    file_list = os.listdir(folder)
    for cot_result_path in file_list:
        if 'cot' in cot_result_path:
            cot_result_path = os.path.join(folder,cot_result_path)
        else:
            continue
        print(cot_result_path)
        with open(cot_result_path,'r',encoding='utf-8') as f:
            pred_lines = f.readlines()
        print("all data length",len(pred_lines))
        if len(pred_lines) < 1:
            continue

        id2cot= {}
        for pred_line in pred_lines:
            product_id = pred_line.strip().split()[0]
            cot = (" ").join(pred_line.strip().split()[1:])
            cot = cot.replace("$","\n")
            
            # product_id = product_id.split("cot_test_")[-1]
            
           
            id2cot[product_id] = cot

        
        





    truth_path = "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/data/test_data/sft_test.txt"
    with open(truth_path,'r',encoding='utf-8') as f:
        lines = f.readlines()
    all_json_data = []
    import copy
    for line in lines:
        line = json.loads(line)
        id = line["id"]
        if id in id2cot:
            line["label"] = copy.deepcopy(line["conversations"][1]["value"])
            line["conversations"][1]["value"] = id2cot[id]
            all_json_data.append(line)

        # print(label,cate_label)
    with open(f"{folder}/cot_result_trans.json",'r',encoding="utf-8") as f:
        for line in all_json_data:
            f.write(json.dumps(line,ensure_ascii=False))
            f.write("\n")


folder = "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_2/m_cot_ckpt/checkpoint-768"
process_cot_result(folder)
   





