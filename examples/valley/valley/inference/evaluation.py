import re
from sklearn.metrics import classification_report
import pandas as pd

def label_category(label):
    import re
 
   
    if   "否" in label:
        return "负例"
    if bool(re.search(r'[a-zA-Z]+\d', label)):  # 字母+数字
        return "禁售"
    elif bool(re.search(r'\d+\.\d', label)):  # 数字.数字
        return "类目错放"
    elif bool(re.search(r'[\u4e00-\u9fa5A-Za-z]+', label)):  # 文字
        return "虚假宣传"
    else:
        print(label)
        raise ValueError("aa")
        


def evaluate_performance(y_true, y_pred):
    # y_true_category = [label_category(y) for y in y_true]
    # y_pred_category = [label_category(y) for y in y_pred]
    categories = [ "虚假宣传", "空值", "类目错放","禁售","负例"]
    y_true = pd.Categorical(y_true, categories=categories, ordered=True)
    y_pred = pd.Categorical(y_pred, categories=categories, ordered=True)
    jinshou_dict = {}

    for true,pred in zip(y_true,y_pred):
        if pred == "禁售":
            if true not in jinshou_dict:
                jinshou_dict[true] = 0
            jinshou_dict[true] += 1
    
    print(jinshou_dict)

    result = classification_report(y_true, y_pred)
    print(result)
   

format_1_file = [
    # "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_ckpt/checkpoint-3000/sft_result.txt",
    "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_ckpt/checkpoint-6000/new_sft_result.txt",
    # # "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_cate_ckpt/checkpoint-4000/new_sft_result.txt",
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_cate_ckpt/checkpoint-8000/new_sft_result.txt",
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_all_prefix_ckpt/checkpoint-8000/new_sft_result.txt",
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_all_prefix_ckpt/checkpoint-16000/new_sft_result.txt",
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_group_ckpt/checkpoint-6000/new_sft_result.txt",
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_info_gen_ckpt/checkpoint-5000/new_sft_result.txt",

]

format_2_file = [
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_group_ckpt/checkpoint-12000/sft_result.txt",
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_info_gen_ckpt/checkpoint-10000/sft_result.txt",
    # "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_all_ckpt/ checkpoint-8000"
    #  "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_ckpt/checkpoint-5000/sft_result.txt",
    "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_from_m_cot_only_ckpt/checkpoint-4686/sft_result.txt",
    "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_from_all_cot8898_ckpt/checkpoint-6000/sft_result.txt"
]
import json
truth_path = "/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/data/test_data/sft_test.txt"
with open(truth_path,'r',encoding='utf-8') as f:
    lines = f.readlines()
truth_data_ids = []
truth_data_labels = []
for line in lines:
    line = json.loads(line)
    id = line["id"]
    label = line["conversations"][1]["value"]
    cate_label = label_category(label)
    truth_data_ids.append(id)
    truth_data_labels.append(cate_label)
    # print(label,cate_label)

truth_df = pd.DataFrame({"truth_label":truth_data_labels,"product_id":truth_data_ids})
print(truth_df["truth_label"].value_counts())

    


for pred_path_1 in format_1_file:
    print(pred_path_1)
    with open(pred_path_1,'r',encoding='utf-8') as f:
        pred_lines = f.readlines()
    print("all data length",len(pred_lines))
    pred_labels,product_ids = [],[]
    for pred_line in pred_lines:
        pred_label,product_id = pred_line.strip().split('\t')
        pred_label = pred_label.replace('#','')
        cate_pred_label = label_category(pred_label)
        # print(pred_label,cate_pred_label)
        pred_labels.append(cate_pred_label)
        product_ids.append(product_id)

    pred_df = pd.DataFrame({"pred_label":pred_labels,"product_id":product_ids})
    all_data_df = pd.merge(truth_df,pred_df,on="product_id",how="left")
    all_data_df["truth_label"] = all_data_df["truth_label"].fillna("空值")
    all_data_df["pred_label"] = all_data_df["pred_label"].fillna("空值")
    
    # print(all_data_df)
    evaluate_performance(y_true = all_data_df["truth_label"], y_pred=all_data_df["pred_label"])
theta_dict = {"虚假宣传":0.55,"类目错放":0.56,"禁售":0.5}
for pred_path_2 in format_2_file:
    print(pred_path_2)
    with open(pred_path_2,'r',encoding='utf-8') as f:
        pred_lines = f.readlines()
    print("all data length",len(pred_lines))
    pred_labels,product_ids = [],[]
    for pred_line in pred_lines:
        product_id,true_label,score,pred_label = pred_line.strip().split('\t')
        cate_pred_label = label_category(pred_label)
        if cate_pred_label in theta_dict:
            if float(score) < theta_dict[cate_pred_label]:
                cate_pred_label = "负例"
        # print(pred_label,cate_pred_label)
        pred_labels.append(cate_pred_label)
        product_ids.append(product_id)

    pred_df = pd.DataFrame({"pred_label":pred_labels,"product_id":product_ids})
    all_data_df = pd.merge(truth_df,pred_df,on="product_id",how="left")
    all_data_df["truth_label"] = all_data_df["truth_label"].fillna("空值")
    all_data_df["pred_label"] = all_data_df["pred_label"].fillna("空值")
    
    # print(all_data_df)
    evaluate_performance(y_true = all_data_df["truth_label"], y_pred=all_data_df["pred_label"])





