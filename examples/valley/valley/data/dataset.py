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
        # print(_url)
        _url = get_url(_url)
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
        
        list_data_dict = json.load(open(data_path, "r")) if data_path else []
        if data_args.video_data_path is None:
            list_video_data_dict = []
        elif os.path.isfile(data_args.video_data_path):
            list_video_data_dict = json.load(open(data_args.video_data_path, "r")) if data_args.video_data_path else []
        else:
            list_video_data_dict = []
            video_data_path_list = os.listdir(data_args.video_data_path)
            for file_name in tqdm(video_data_path_list):
                data_path = os.path.join(data_args.video_data_path, file_name)
                list_video_data_dict += json.load(open(data_path, "r"))
        list_data_dict = list_video_data_dict + list_data_dict
        random.shuffle(list_data_dict)
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.inference = inference
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
        try:
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            if ('image' in sources[0]) and isinstance(self.list_data_dict[i]['image'], str):       ### for single image
                print(1)
                image_file = self.list_data_dict[i]['image']
                image_folder = self.data_args.image_folder
                if 'train2014' in image_folder:
                        image_file = 'COCO_train2014_'+image_file
                processor = self.data_args.image_processor
                image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')

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
                print(2)
                image_list = []
                for image_file in self.list_data_dict[i]['image'][:self.data_args.max_img_num]:
                    # image_folder = self.data_args.image_folder if self.data_args.image_folder else ''
                    image_str = download_url_with_exception(image_file)
                    image = Image.open(io.BytesIO(image_str)).convert('RGB')
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
                print(3)
                video_file = self.list_data_dict[i]['video']
                processor = self.data_args.image_processor
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
                print(4)
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
            # image exist in the data
            if 'image' in self.list_data_dict[i] or 'video' in self.list_data_dict[i]:
                data_dict['image'] = image
            elif self.data_args.is_multimodal:
                # image does not exist in the data, but the model is multimodal
                crop_size = self.data_args.image_processor.crop_size
                data_dict['image'] = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
            if 'gt_label' in self.list_data_dict[i]:
                data_dict['gt_label'] = self.list_data_dict[i]['gt_label']
            return data_dict
        except Exception as e:
            traceback.print_exc()
            print(self.list_data_dict[i]['id'])
            print(e)
            return ('fail', sources)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances_no_error = []
        for ins in instances:
            if type(ins) != tuple and len(ins["input_ids"]) < self.tokenizer.model_max_length:
                instances_no_error.append(ins)
        instances = instances_no_error
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
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
            gt_label = [instance['gt_label'] for instance in instances]
            batch['gt_label'] = gt_label
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
