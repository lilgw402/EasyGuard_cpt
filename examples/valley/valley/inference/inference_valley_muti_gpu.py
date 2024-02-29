import argparse
from cgi import test
import torch
import sys
sys.path.append("/opt/tiger/EasyGuard/examples/valley")
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from valley.model.language_model.valley_llama import ValleyVideoLlamaForCausalLM, ValleyProductLlamaForCausalLM
import torch
import os
from valley.utils import disable_torch_init
import os
import random
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from torch.utils.data.distributed import DistributedSampler
from valley.util.config import DEFAULT_GANDALF_TOKEN
from valley.util.data_util import KeywordsStoppingCriteria
from peft import PeftConfig
from transformers import set_seed
from valley.data.dataset_cp import LazySupervisedDataset, DataCollatorForSupervisedDataset
from valley.util.data_util import smart_tokenizer_and_embedding_resize
from valley import conversation as conversation_lib
from valley.util import ddp_utils as utils

os.environ['NCCL_DEBUG'] = ''


def standardization(data):
    mu = torch.mean(data)
    sigma = torch.std(data)
    return (data - mu) / sigma


def inference(args):
    set_seed(42)
    print(args)
    utils.init_distributed_mode(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    world_size = utils.get_world_size()
    rank = utils.get_rank()

    disable_torch_init()

    Model = None
    if args.model_class == 'valley-video':
        Model = ValleyVideoLlamaForCausalLM
    elif args.model_class == 'valley-product':
        Model = ValleyProductLlamaForCausalLM

    model_name = os.path.expanduser(args.model_name)

    # load model
    # if 'lora' in model_name:
    if args.lora:
        config = PeftConfig.from_pretrained(model_name)
        print('load old model weight and lora weight')
        model_old = Model.from_pretrained(model_name, torch_dtype=torch.float16)
        print('load no lora model')
        if os.path.exists(os.path.join(model_name, 'non_lora_trainables.bin')):
            non_lora_state_dict = torch.load(os.path.join(model_name, 'non_lora_trainables.bin'))
            new_state_dict = dict()
            for key in non_lora_state_dict.keys():
                key_new = '.'.join(key.split('.')[2:])  # base_model.model.model.xxxx
                new_state_dict[key_new] = non_lora_state_dict[key]
            model_old_state = model_old.state_dict()
            model_old_state.update(new_state_dict)
            model_old.load_state_dict(model_old_state)
        model = model_old
        tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(model_name), use_fast=False)
        if hasattr(model.model, 'gandalf_projector'):
            model_old.config.gandalf_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_GANDALF_TOKEN)
        tokenizer.padding_side = 'left'
        print("load end")
    else:
        print('load model')
        model = Model.from_pretrained(
            model_name, torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, use_fast=False)
        tokenizer.padding_side = 'left'
        print('load end')

    if args.language == 'chinese':
        from transformers import ChineseCLIPImageProcessor as CLIPImageProcessor
    else:
        from transformers import CLIPImageProcessor

    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
    model.eval()
    model = model.to(device)

    args.image_processor = image_processor
    args.is_multimodal = True
    args.mm_use_im_start_end = True
    args.only_mask_system = False

    if args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if args.prompt_version is not None:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]
    dataset = LazySupervisedDataset(args.data_path, tokenizer=tokenizer, data_args=args, inference=True)

    if args.DDP:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset,
                                pin_memory=True, sampler=sampler, )

        output_folder = os.path.dirname(args.out_path)
        os.makedirs(output_folder, exist_ok=True)

        rf = open(args.out_path + ".worker_" + str(rank), 'w')
    else:
        dataloader = DataLoader(dataset, num_workers=1, batch_size=args.batch_size,
                                collate_fn=DataCollatorForSupervisedDataset, pin_memory=True)
        rf = open(args.out_path, 'w')

    prog_bar = tqdm(dataloader, total=len(dataloader), desc='worker_' + str(rank)) if rank == 0 else dataloader

    for test_batch in prog_bar:
        try:
            test_batch = test_batch.tokenizer[0]
            gt_label = test_batch["gt_label"]
            id = test_batch["product_id"]
            if not isinstance(gt_label,list):
                gt_label = [gt_label]
            if not isinstance(id,list):
                id = [id]
            for key in test_batch:
                if key not in ['product_id','gt_label']:
                    test_batch[key] = test_batch[key].to(device)
            # for key in test_batch:
            #     test_batch[key] = test_batch[key].to(device)
            stop_str = conversation_lib.default_conversation.sep if conversation_lib.default_conversation.sep_style != conversation_lib.SeparatorStyle.TWO else conversation_lib.default_conversation.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, test_batch['input_ids'].unsqueeze(0))
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids=test_batch['input_ids'].unsqueeze(0),
                    images=test_batch['image'].half().unsqueeze(0),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    stopping_criteria=[stopping_criteria],
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True if args.ouput_logits else False,
                    output_scores=True if args.ouput_logits else False
                )
            if not args.ouput_logits:
                input_token_len = test_batch['input_ids'].unsqueeze(0).shape[1]
                n_diff_input_output = (
                            test_batch['input_ids'].unsqueeze(0) != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)

            if args.ouput_logits:
                input_token_len = test_batch['input_ids'].unsqueeze(0).shape[1]
                n_diff_input_output = (test_batch['input_ids'].unsqueeze(0) != output_ids.sequences[:,
                                                                               :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(
                        f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode(output_ids.sequences[:, input_token_len:], skip_special_tokens=True)

                scores = standardization(output_ids.scores[3])
                yes_score, no_score = scores[0, 34043], scores[0, 29871]  # scores[0,31191]
                standardization_score = torch.stack([yes_score, no_score])

                yes_logits = torch.softmax(standardization_score, dim=0)[0].cpu().numpy().tolist()
                yes_logits = format(yes_logits, '.6f')

                generated_scores = []
                for j in range(len(outputs)):
                    generated_scores.append(yes_logits)

            for i, out in enumerate(outputs):
                while True:
                    cur_len = len(out)
                    out = out.strip()
                    for pattern in ['###', '##', 'Assistant:', 'Response:', 'LLaVA:', '助理']:
                        if out.startswith(pattern):
                            out = out[len(pattern):].strip()
                    if len(out) == cur_len:
                        break
                out = out.replace('\n###', '').replace('\n##', '').strip()

                if not args.ouput_logits:
                    out = out.replace("\n","$")
                    res = [str(id[i]), out]
                else:
                    res = [str(id[i]), str(gt_label[i]), str(generated_scores[i]), out]
                print(res)
                rf.write('\t'.join(res) + '\n')
                rf.flush()

        except Exception as e:
            traceback.print_exc()
    rf.close()


def gather_result(args):
    dist.barrier()
    rank = utils.get_rank()
    num_worker = utils.get_world_size()
    if rank == 0:
        with open(args.out_path, 'a+',encoding="utf-8") as f:
            for i in range(num_worker):
                with open(args.out_path + ".worker_" + str(i), 'r',encoding="utf-8") as tf:
                    tmp_result = tf.readlines()
                f.writelines(tmp_result)
                os.remove(args.out_path + ".worker_" + str(i))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_class", type=str, default="valley-product")
    parser.add_argument("--language", type=str, default="chinese")
    parser.add_argument("--model_name", type=str, default = '/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_ckpt/checkpoint-3000')
    parser.add_argument("--video_data_path", type=str, required = False, default = None)
    parser.add_argument("--data_path", type=str, required = False, default = '/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/data/test_data/sft_test.txt' )
    parser.add_argument("--video_folder", type=str, required = False, default = None)
    parser.add_argument("--image_folder", type=str, required = False, default = '/mnt/bn/valley2/hezhongheng/mllm_image/continut_pretrain/')
    parser.add_argument("--out_path", type=str, required = False, default = '/mnt/bn/valley2/hezhongheng/continue_pretrain_multi/ckpt/v_2_1/sft_ckpt/checkpoint-3000/sft_result.txt' )
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--prompt_version", type=str, default="conv_prd_cp")
    parser.add_argument("--max_img_num", type=int, default=8)
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--ouput_logits", action="store_true", default=False)
    parser.add_argument("--temperature", type = float, default=1)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument("--DDP_port", default = '12345')
    parser.add_argument("--world_size", type=int, default = 1)
    parser.add_argument("--lora", default = True)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--max_new_tokens', type=int, default=150)


   
    args = parser.parse_args()

    inference(args)
    gather_result(args)