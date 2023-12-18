import os
import csv
import json, re
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score, f1_score


datas = {}
first_types = set()
with open('/mnt/bn/yangmin-priv-fashionmm/Data/sk/continue_data/shouyi/zhunru_test.txt') as f:
    for line in f:
        jsd = json.loads(line.strip())
        pid = jsd['product_id']
        # datas.append(jsd)
        datas.setdefault(pid, [])
        datas[pid].append(jsd)
        first_type = jsd.get('first_type', '')
        if first_type:
            first_types.add(jsd['first_type'])
print(first_types)

        

def get_zhunru_result(path, t='高危禁售'):

    y_true = []
    y_pred = []



    used_pids = set()
    with open(path) as f:
        for i, line in enumerate(f):
            tokens = line.strip().split('\t')
            # print(tokens)
            model_ans = re.sub('Assistant: ', '', tokens[0])
            pid = tokens[1]

            if pid in used_pids:
                continue
            used_pids.add(pid)
            
            for jsd in datas[pid]:
                # if value in jsd['conversations'][0]['value'] and value in ques:
                f = jsd.get('first_type', '')

                if not f or t in f:
                    pred = 0
                    if model_ans.startswith('不'):
                        pred = 1
                    
                    label = 0
                    ans = jsd['conversations'][1]['value']
                    if ans.startswith('不'):
                        label = 1
                    
                    y_pred.append(pred)
                    y_true.append(label)

    labels = ['通过', t]
    report = classification_report(y_true=y_true, y_pred=y_pred, target_names=labels)
    print(report)

def get_zhunru_res(path):
    t_list = ['高危禁售', '中危禁售', '底线问题', '类目错放', '站外引流', '虚假宣传']
    for t in t_list:
        get_zhunru_result(path, t)

get_zhunru_res('/mnt/bn/yangmin-priv-fashionmm/Data/sk/continue_data/cp_mllm_output_basic/cp_7b_lora_pool_crop336_down.txt')