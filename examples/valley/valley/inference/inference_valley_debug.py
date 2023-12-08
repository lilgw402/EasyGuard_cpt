import argparse
import torch
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
from valley.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from valley.util.data_util import smart_tokenizer_and_embedding_resize
from valley import conversation as conversation_lib
os.environ['NCCL_DEBUG']=''
def setup(args,rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.DDP_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size, )



def inference(rank, world_size, args):
    set_seed(42)

    this_rank_gpu_index = rank

    if args.DDP:
        torch.cuda.set_device(this_rank_gpu_index)
        setup(args, rank, world_size)
        
    disable_torch_init()

    device = torch.device('cuda:'+str(this_rank_gpu_index)
                          if torch.cuda.is_available() else 'cpu')

    Model = None
    if args.model_class == 'valley-video':
        Model = ValleyVideoLlamaForCausalLM
    elif args.model_class == 'valley-product':
        Model = ValleyProductLlamaForCausalLM

    model_name = os.path.expanduser(args.model_name)


    tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(model_name), use_fast = False)


    tokenizer.padding_side = 'left'


    args.image_processor = image_processor
    args.is_multimodal = True
    args.mm_use_im_start_end = True
    args.only_mask_system = False
    dataset = LazySupervisedDataset(args.data_path, tokenizer=tokenizer, data_args = args, inference= True)
    
    if args.DDP:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset, pin_memory=True, sampler=sampler,)
        rf = open(args.out_path+".worker_"+str(rank), 'w')
    else:
        dataloader = DataLoader(dataset, num_workers=1, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset, pin_memory=True)
        rf = open(args.out_path, 'w')

    prog_bar = tqdm(dataloader, total=len(dataloader),desc='worker_'+str(rank)) if rank == 0 else dataloader
    

    for test_batch in prog_bar:
        test_batch = test_batch.tokenizer[0]
        # gt_label = [test_batch.pop('gt_label')]
        for key in test_batch:
            print(key, test_batch[key])
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-class", type=str, default="valley-product")
    parser.add_argument("--language", type=str, default="chinese")
    parser.add_argument("--model-name", type=str, default = '/mnt/bn/yangmin-priv-fashionmm/Data/sk/checkpoints/valley-chinese-7b-lora-product-continue-pretrain-down-pool-5epoch-v2/checkpoint-12000')
    parser.add_argument("--video_data_path", type=str, required = False, default = None)
    parser.add_argument("--data_path", type=str, required = False, default = '/mnt/bn/yangmin-priv-fashionmm/Data/sk/continue_data/shouyi/zhunru_test.json' )
    parser.add_argument("--video_folder", type=str, required = False, default = None)
    parser.add_argument("--image_folder", type=str, required = False, default = '/mnt/bn/yangmin-priv-fashionmm/projects/zhaoziwang/data/chinese_valley_test_image/image/')
    parser.add_argument("--out_path", type=str, required = False, default = '/mnt/bn/yangmin-priv-fashionmm/Data/sk/vulgar/data/valley_v1data_without_ocr_eval_res_step2000_debug_easyguard_v2.txt' )
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--prompt_version", type=str, default="conv_prd_cp")
    parser.add_argument("--max_img_num", type=int, default=8)
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--ouput_logits", action="store_true", default=False)
    parser.add_argument("--temperature", type = float, default=1)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument("--DDP_port", default = '12345')
    parser.add_argument("--world_size", type=int, default = 1)
    args = parser.parse_args()

    mp.spawn( inference, args=(args.world_size, args), nprocs=args.world_size)