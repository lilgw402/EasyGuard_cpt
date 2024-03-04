
import os
import json
caption_data_folder = "/mnt/bn/yangmin-priv-fashionmm/wangzhen/data/caption"
all_caption_data_path = "/mnt/bn/yangmin-priv-fashionmm/Data/zhongheng/continue_pretrain_multi/data/train_data/wxl/vision_vocab/train_data_1/train.jsonl"
all_caption_data = []
file_name_list = os.listdir(caption_data_folder)
print(file_name_list)

for file_name in file_name_list:
    print(file_name)
    if "jsonl" in file_name:
        with open(f'{caption_data_folder}/{file_name}','r',encoding="utf-8") as f:
            all_caption_data.extend(f.readlines())

print("all caption data:",len(all_caption_data))

wf = open(all_caption_data_path,'w',encoding="utf-8")
for caption in all_caption_data:
    caption = json.loads(caption)
    wf_content = {
        "image":caption["image"],
         "conversations": [
             {"from": "human", "value": ""},
            {"from": "gpt", "value": caption["caption"]}
            ]
    }
    wf.write(json.dumps(wf_content,ensure_ascii=False))
    wf.write("\n")
wf.close()