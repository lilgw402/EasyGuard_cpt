import json
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import random
import os
import torch
import json
import transformers
from typing import Dict, Sequence
from dataclasses import dataclass
from valley.util.config import *
from valley.util.data_util import preprocess, preprocess_multimodal
import copy
import random
import numpy as np
from torchvision import transforms
import io
import decord
import traceback
import urllib

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def get_url(url):
    if 'http' in url:
        url = url
    elif 'v1_' in url:
        url = 'https://tosv.byted.org/obj/ecom-shop-material/' + url
    elif url.find('-image.image') != -1:
        url = 'http://p3-im.byteimg.com/tos-cn-i-scl3phc04j/' + url
    elif url.find('.image') != -1:
        url = 'http://tosv.byted.org/obj/temai/' + url
    elif url.find('.jpeg') != -1:
        url = 'http://p3-im.byteimg.com/tos-cn-i-scl3phc04j/' + url.replace('jpeg', 'image')
    elif url.find('.png') != -1:
        url = 'http://p3-im.byteimg.com/tos-cn-i-scl3phc04j/' + url.replace('png', 'image')
    else:
        url_list = url.split('/')[-1].split('_')
        if len(url_list)>4:
            url = 'http://tosv.byted.org/obj/ecom-shop-material/' + url
        else:
            url = 'http://tosv.byted.org/obj/temai/' + url
    if url.find('~720x0.image')!=-1:
        url = url.replace('~720x0.image','')
    url = url.replace('p9-aio.ecombdimg.com','tosv.byted.org').replace('p6-aio.ecombdimg.com','tosv.byted.org').replace('p3-aio.ecombdimg.com','tosv.byted.org')
    return url

def download_url_with_exception(_url, timeout=5):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
        }
        _url = get_url(_url)
        # print('_url',_url )
        req = urllib.request.Request(url=_url, headers=headers)
        response = urllib.request.urlopen(req, timeout=timeout)
        return response.read()
    except Exception as e:
        print('download error', e)
        return b''

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args,
                 inference):
        super(LazySupervisedDataset, self).__init__()
        
        list_data_dict = []
        breakpoint()
        if os.path.isfile(data_path) and data_path[-4:] != 'json':
            list_data_dict = [json.loads(data) for data in open(data_path, 'r').readlines()]
        else:
            list_data_dict = json.load(open(data_path, "r"))
        print(list_data_dict[:2])

        #对于视频数据
        
        if data_args.video_data_path is None: #True
            list_video_data_dict = []
        elif os.path.isfile(data_args.video_data_path):
            list_video_data_dict = json.load(open(data_args.video_data_path, "r")) if data_args.video_data_path else []
        else:
            list_video_data_dict = []
            video_data_path_list = os.listdir(data_args.video_data_path)
            for file_name in tqdm(video_data_path_list):
                data_path = os.path.join(data_args.video_data_path, file_name)
                list_video_data_dict += json.load(open(data_path, "r"))
        #之后将视频数据字典列表 `list_video_data_dict` 和 `list_data_dict` 合并
        list_data_dict = list_video_data_dict + list_data_dict
        random.shuffle(list_data_dict)
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.inference = inference
        self.image_path = data_args.image_folder
        print('-------image_path-------:', self.image_path) #/mnt/bn/valley2/hezhongheng/mllm_image/continut_pretrain

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        print("sources---------------------",sources)
        try:
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            if ('image' in sources[0]) and isinstance(self.list_data_dict[i]['image'], str):       ### for single image
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                if 'train2014' in image_folder:
                        image_file = 'COCO_train2014_'+image_file
                processor = self.data_args.image_processor
                # image = Image.open(os.sources = copy.deepcopyath.join(image_folder, image_file)).convert('RGB')
                image_str = download_url_with_exception(image_file)
                image = Image.open(io.BytesIO(image_str)).convert('RGB')

                if self.data_args.image_aspect_ratio == 'pad':
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                #image shape [3,336,336]
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
                image = image.unsqueeze(0)
            elif ('image' in sources[0]) and isinstance(self.list_data_dict[i]['image'], list):     ### for multi image 
                image_list = []
                # print('image path: ')
                for image_file in self.list_data_dict[i]['image'][:self.data_args.max_img_num]:
                    image_folder = self.data_args.image_folder if self.data_args.image_folder else ''
                    try:
                        save_name = image_file
                        image_path = os.path.join(self.image_path, save_name + '.png')
                        # print(image_path)
                        if os.path.exists(image_path):
                            image = Image.open(image_path).convert('RGB')
                        else:   
                            # print(image_file)
                            image_str = download_url_with_exception(image_file)
                            image = Image.open(io.BytesIO(image_str)).convert('RGB')

                    except Exception as e:
                        # print(e)
                        image_str = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xdb\x00C\x01\t\t\t\x0c\x0b\x0c\x18\r\r\x182!\x1c!22222222222222222222222222222222222222222222222222\xff\xc0\x00\x11\x08\x01\x00\x01\x00\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xc4\x00\x1f\x01\x00\x03\x01\x01\x01\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x11\x00\x02\x01\x02\x04\x04\x03\x04\x07\x05\x04\x04\x00\x01\x02w\x00\x01\x02\x03\x11\x04\x05!1\x06\x12AQ\x07aq\x13"2\x81\x08\x14B\x91\xa1\xb1\xc1\t#3R\xf0\x15br\xd1\n\x16$4\xe1%\xf1\x17\x18\x19\x1a&\'()*56789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00?\x00\xf9\xfe\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x02\x8a(\xa0\x0f\xff\xd9'
                        image = Image.open(io.BytesIO(image_str)).convert("RGB")
                    processor = self.data_args.image_processor
                    # image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                    else:
                        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

                    image_list.append(image)
                image_list =  torch.stack(image_list, dim = 0)
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]),
                    self.data_args)
                image = image_list
            elif 'video' in sources[0]:                                                     ### for video file or folder
                video_file = self.list_data_dict[i]['video']
                processor = self.data_args.image_processor
                # print(processor)
                if 'source' not in self.list_data_dict[i]:
                    video_file = os.path.join(self.data_args.video_folder, video_file)
                else:
                    video_file_source = self.list_data_dict[i]['source']
                    video_file = os.path.join(self.data_args.video_folder, video_file_source, video_file)
                
                if os.path.isfile(video_file):
                    video_reader = decord.VideoReader(video_file, num_threads=1, ctx= decord.cpu(0))
                    decord.bridge.set_bridge('torch')
                    video_len = len(video_reader)
                    video = video_reader.get_batch(np.linspace(0, video_len - 1, 8).astype(np.int_)).byte()  # 8, height,width,3
                else:
                    if os.path.exists(video_file):
                        video = [os.path.join(video_file, file) for file in os.listdir(video_file)][:self.data_args.max_img_num]
                    else:
                        video = []
                    padded_list = ['/mnt/bn/zhaoziwang/multimodal-pretrain-data/demodata/blackimage/black_image.png']*max(8-len(video),0) # this 
                    video = video + padded_list
                video_pad = []
                for image in video:
                    if isinstance(image, str):
                        imagetoPIL = Image.open(image)
                    else:
                        imagetoPIL = transforms.ToPILImage()(image.permute(2,0,1)).convert('RGB')
                    
                    if self.data_args.image_aspect_ratio == 'pad':
                        imagetoPIL = expand2square(imagetoPIL, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(imagetoPIL, return_tensors='pt')['pixel_values'][0]
                    video_pad.append(image)
                video = torch.stack(video_pad, dim = 0)
                sources = preprocess_multimodal(
                        copy.deepcopy([e["conversations"] for e in sources]),
                        self.data_args)
                image = video
            else:
                sources = copy.deepcopy([e["conversations"] for e in sources])
            if self.inference and len(sources[0])%2 == 0:
                sources[0] = sources[0][:-1]
            data_dict = preprocess(
                sources,
                self.tokenizer,
                has_image=('image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]),
                only_mask_system= self.data_args.only_mask_system,
                inference = self.inference)
            
            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
            
            # labels = data_dict['labels'].tolist()
            # labels_ = [a for a in labels if a!= -100]
            # print('labels', labels_)
            # back_str = self.tokenizer.convert_ids_to_tokens(labels_)
            # print('back_str', back_str)

            # input_ids = data_dict['input_ids']
            # input_ids_ = [a for a in input_ids if a!= -200]
            # print('input_ids', input_ids)
            # back_str = self.tokenizer.convert_ids_to_tokens(input_ids_)
            # print('back_str', back_str)
            # print('imgae_size', image.shape)

            # image exist in the data
            # print(self.list_data_dict[i].keys())
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
            if 'gt_label' in self.list_data_dict[i]:
                data_dict['gt_label'] = self.list_data_dict[i]['gt_label']
            if 'label' in self.list_data_dict[i]:
                data_dict['gt_label'] = self.list_data_dict[i]['label']
            if 'product_id' in self.list_data_dict[i]:
                data_dict['product_id'] = self.list_data_dict[i]['product_id']
            if 'id' in self.list_data_dict[i]:
                data_dict['product_id'] = self.list_data_dict[i]['id']
            print("data_dict===============",data_dict)
            # data_dict['source'] = sources
            return data_dict
        except Exception as e:
            traceback.print_exc()
            # print(self.list_data_dict[i]['id'])
            print(e)
            return ('fail', sources)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances_no_error = []
        for ins in instances:
            if type(ins) == tuple:
                continue
           
            ins["input_ids"] = ins["input_ids"][:self.tokenizer.model_max_length]
            ins["labels"] = ins["labels"][:self.tokenizer.model_max_length]
            # print("label_length",len(ins["labels"]))
            # print("input_ids_length",len(ins["input_ids"]))

            if type(ins) != tuple and len(ins["input_ids"]) <= self.tokenizer.model_max_length:
                instances_no_error.append(ins)
        if len(instances_no_error) < 1:
            print("instance_count",len(instances_no_error))
        instances = instances_no_error
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        # print(len(input_ids))
        # if len(input_ids) > 0:
        #     print(type(input_ids))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'gt_label' in instances[0]:
            try:
                gt_label = [instance['gt_label'] for instance in instances]
                batch['gt_label'] = gt_label
            except:
                pass
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, inference = False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                inference = inference)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


if __name__ == '__main__':

    from valley.train.train import ModelArguments, DataArguments, TrainingArguments
    from valley.model.language_model.valley_llama import ValleyProductLlamaForCausalLM
    from valley.util.data_util import smart_tokenizer_and_embedding_resize
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file('./valley/configs/experiment_cn/valley_cn_7b_product_cp_hzh_v1_2.yaml')

    tokenizer = transformers.LlamaTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    bnb_model_from_pretrained_args = {}
    model = ValleyProductLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
        )

    data_args.model_class = model_args.model_class

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model,
        )

    model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    if model_args.mm_vision_select_feature == 'cls_patch':
        vision_tower.select_feature = 'cls_patch'
    data_args.image_processor = vision_tower.image_processor
    print('vision_tower.image_processor', data_args.image_processor)
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    train_dataset = data_module["train_dataset"]

    it = iter(train_dataset)
    data = next(it)
    print(data)

    

