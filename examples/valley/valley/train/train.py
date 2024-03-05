import pathlib
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List
import transformers
from valley.train.trainner import  ValleyTrainer
from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from valley.model.language_model.valley_llama import ValleyVideoLlamaForCausalLM, ValleyProductLlamaForCausalLM
from valley.model.language_model.valley_mistral import ValleyMistralForCausalLM
from valley.util.config import *
from valley.util.data_util import load_video
from tqdm import tqdm
import copy
import logging
import json
import random
from dataclasses import dataclass, field
from typing import Optional
import os
from valley.utils import print_trainable_params
from PIL import Image
import decord
import warnings
from valley import conversation as conversation_lib
import numpy as np
from torchvision import transforms
from valley.data import video_transform
import random
import argparse
from torch import nn
import traceback
from valley.util.data_util import smart_tokenizer_and_embedding_resize
from valley.data.dataset_cp import make_supervised_data_module
local_rank = None
os.environ['NCCL_DEBUG']=''

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

warnings.filterwarnings("ignore", category=FutureWarning)

@dataclass
class ModelArguments:
    model_class: Optional[str] = field(default="valley")
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    lora_model:  Optional[str] = None
    version: Optional[str] = field(default="v0")
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None) #预训练的多层感知器适配器的路径。
    mm_use_im_start_end: bool = field(default=False)
    tune_llm_layer: str=field(default= None) #指定大语言模型的调整
    patch_pooling_method: str=field(default='mean')
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="cls_patch")
    tune_mm_mlp_adapter: bool = field(default=False)
    language: Optional[str] = field(default='xl')
    pool_out_size: int = field(default=8)
    image_crop_width: int = field(default=224)
    image_crop_height: int = field(default=224)
    
@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    video_data_path:str = field(default = None,
                            metadata={"help": "Path to the video training data."})
    is_fashion_data: str = field(default = False,
                            metadata={"help": "wether bussiness data"})
    max_img_num : int = field(default = 8,
                            metadata={"help": "maximum number of images in one conversation."})
    lazy_preprocess: bool = False
    is_multimodal: bool = True
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    fast_epoch: bool = field(default=False)
    conv_mode:str = field(default = 'v1')
    project_name: str = field(default='valley')
    frame_mode: str = field(default = 'fixed')
    fixed_frame_number: int = field(default = 8)
    fps: float=field(default=0.5)
    image_grid_pinpoints: Optional[str] = field(default=None)
    prompt_version: str = None
    only_mask_system: bool =  False
    use_gandalf_vector: bool = False

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False) #决定是否冻结模型的特定部分。
    freeze_backbone: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    #与模型量化相关的参数
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    #LoRA相关设置
    lora_enable: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_merge: bool = False
    lora_save_strategy: str = 'no'
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

#模型中的参数移动到 CPU，并深拷贝，以防被分布式训练库（如Deepspeed的Zero）清空
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param




# 根据拥有 "lora_" 或 "bias" 前缀的策略选择参数并处理：将选中的参数移动到 CPU。策略不同，选择参数的范围也不同。
# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return

#选择模型中没有 "lora_" 前缀的参数，使用 `maybe_zero_3` 处理并选择参数。
def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

#选择模型中匹配特定关键字的参数，参数被选中后，通过 `maybe_zero_3` 保存到 CPU。
def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

#在模型架构中找到所有的 `torch.nn.Linear` 模块，遍历模型的所有子模块，并排除所有包含指定关键字的模块
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # print(name)
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    # print(list(lora_module_names))
    return list(lora_module_names)

#安全地保存使用 Hugging Face `Trainer` 类训练的模型
def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


#回调是机器学习训练过程中用来定义和执行在特定时刻发生的行为的一种工具。在Hugging Face的训练器（Trainer）中，你可以定义自己的回调来增强训练过程或包含额外的日志、保存逻辑等
class LLMCallback(TrainerCallback):
    "A callback that output infomation and do some operators"

    
    def output_log(self, args: TrainingArguments, state: TrainerState):
        def loss_log(data):
            try:
                loss_ = data["loss"]
                learning_rate_ = data["learning_rate"]
                step_ = data["step"]
                loss_log_str = f"step: {step_:<8} || learning_rate: {learning_rate_:<25} || loss: {loss_:<10}"
            except:
                loss_log_str = json.dumps(data)
            return loss_log_str

        output_file = os.path.join(args.output_dir, "trainer.log")
        log_history = map(loss_log, state.log_history)
        with open(output_file, "w") as f:
            for line in log_history:
                f.write(line + "\n")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # TODO: support deepspeed zero3 save extra weights not all llm weights
        if args.lora_enable and args.lora_save_strategy == 'steps' and state.global_step%args.save_steps == 0:
            self.output_log(args, state)
            model_ = kwargs["model"]
            save_number = str(state.global_step)
            state_dict = get_peft_state_maybe_zero_3(
                model_.named_parameters(), args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                model_.named_parameters()
            )
            if args.local_rank == 0 or args.local_rank == -1:
                output_dir = os.path.join(args.output_dir,f'checkpoint-{save_number}')
                os.makedirs(output_dir, exist_ok=True)
                model_.config.save_pretrained(output_dir)
                model_.save_pretrained(output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
                kwargs["tokenizer"].save_pretrained(args.output_dir)
        return super().on_step_end(args, state, control, **kwargs)


def load_model(model_args,training_args):

    if model_args.lora_model is not None:
        print("training from lora checkpoint.")
        from peft import PeftConfig,PeftModel

        
        # tokenizer = transformers.LlamaTokenizer.from_pretrained(os.path.dirname(model_args.lora_model), use_fast=False)
        config = PeftConfig.from_pretrained(model_args.lora_model)
        tokenizer = transformers.LlamaTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=False)
        print('load base model')
        model_base = ValleyProductLlamaForCausalLM.from_pretrained(config.base_model_name_or_path, torch_dtype=torch.float16)
       

        if training_args.lora_merge:
            print("lora chekpoint using: Merged.")

            # merge LoRA paramters
            print("load LoRA weight..")
            model_base = PeftModel.from_pretrained(model_base, model_args.lora_model)
            model_merge_lora = model_base.merge_and_unload().half()

            # merge other fituning parameters w/o LoRa
            if os.path.exists(os.path.join(model_args.lora_model, 'non_lora_trainables.bin')):
                print("load non LoRA weight..")
                non_lora_state_dict = torch.load(os.path.join(model_args.lora_model, 'non_lora_trainables.bin'))
                new_state_dict = dict()
                for key in non_lora_state_dict.keys():
                    key_new = '.'.join(key.split('.')[2:])  # base_model.model.model.xxxx
                    new_state_dict[key_new] = non_lora_state_dict[key]

                model_merge_lora_dict = model_merge_lora.state_dict()
                model_merge_lora_dict.update(new_state_dict)
                model_merge_lora.load_state_dict(model_merge_lora_dict)
            
            model = model_merge_lora
            print('load model end')
        
        else:
            model = model_base
            
        

    else:
        print("training from usual model.")
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                model_max_length=training_args.model_max_length,
                padding_side="right",
                use_fast=False,)
        model = ValleyProductLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    )
    return model,tokenizer

        


def train(args):
    breakpoint()
    global local_rank
    #分别定义了模型的结构、训练数据以及训练过程
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_yaml_file(args.conf)
   
    #标志确定训练的精度（`torch.float16`、`torch.bfloat16` 或 `torch.float32`）:torch.float16
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    training_args.learning_rate = float(training_args.learning_rate)
    os.environ['WANDB_PROJECT'] = data_args.project_name
    breakpoint()
    #根据 `model_class` 初始化了词分析器（tokenizer），它可以是 `LlamaTokenizer` 或者 `AutoTokenizer`，并设置了最大模型长度和缓存目录。
    if model_args.model_class in ['valley-video', 'valley-product']: 
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    elif model_args.model_class == 'mistral':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        raise ValueError(f"Unknown Model Class.")

    #为 Bits and Bytes 设置了额外配置，这是 Hugging Face 用于训练低比特权重模型的实验方法    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    
    data_args.model_class = model_args.model_class
    #根据 `vision_tower`，会初始化不同类别的模型，可能是 `ValleyVideoLlamaForCausalLM`、`ValleyMistralForCausalLM` 等。这些模型会加载之前配置的预设设置。
    if model_args.vision_tower is not None:
        if  model_args.model_class == 'valley-video': 
            model = ValleyVideoLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
            )
        elif model_args.model_class == 'mistral':
            model = ValleyMistralForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
            )
        elif model_args.model_class == 'valley-product':
            model, tokenizer = load_model(model_args=model_args, training_args=training_args)
            # model = ValleyProductLlamaForCausalLM.from_pretrained(
            #         model_args.model_name_or_path,
            #         cache_dir=training_args.cache_dir,
            #         **bnb_model_from_pretrained_args
            #         )
            breakpoint()


        else:
            raise ValueError(f"Unknown Model Class.")
    else:
        if model_args.model_class in ['valley-video', 'valley-product']: 
            model = transformers.LlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_class == 'mistral': 
            model = transformers.MistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            raise ValueError(f"Unknown Model Class.")
    breakpoint()        
    #如果设置了 `freeze_backbone`，则会冻结模型中的参数，防止在训练过程中被更新。
    model.config.use_cache = False   
    if training_args.freeze_backbone: #True
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)
    
    #如果启用了 `gradient_checkpointing`，则会修改模型以支持梯度检查点，以节省训练期间的内存
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

   
    #如果启用了LoRA，将会设置LoRA配置，并且根据LoRA配置适配模型
    if training_args.lora_enable: #True
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig( #创建了一个 `LoraConfig` 的实例 `lora_config`，并使用了几个来自 `training_args` 的参数（比如 `lora_r` 和 `lora_alpha`）来配置LoRA的参数
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16: #进行数值类型转换
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)


        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config) #使用LoRA配置更新模型，为模型添加LoRA适配器。


        if model_args.lora_model != None and training_args.lora_merge == False:
            print("lora chekpoint using: Continue Training.")

            peft_model_dict = model.state_dict() #获取当前模型的状态字典


            ## replcae LoRA weight 表明要用LoRA权重替换PEFT模型中的权重
            print("replace peft model with  LoRA weight..")
            #根据LoRA模型的路径加载LoRA参数，创建了一个LoRA参数模型 `LoRA_param_model`
            LoRA_param_model = ValleyProductLlamaForCausalLM.from_pretrained(model_args.lora_model, torch_dtype=torch.float16)
            #将LoRA参数模型的状态字典 `lora_param_dict` 中的参数复制到 `peft_model_dict` 中。
            lora_param_dict = LoRA_param_model.state_dict()
            for key in lora_param_dict.keys():
                new_key = f"base_model.model.{key}"
                assert new_key in peft_model_dict
                peft_model_dict[new_key] = lora_param_dict[key]
            
            ## replcae w/o LoRA weight 如果存在不包含LoRA参数的权重文件 `non_lora_trainables.bin`，执行类似的权重替换操作。
            if os.path.exists(os.path.join(model_args.lora_model, 'non_lora_trainables.bin')):
                print("replace peft model with non LoRA weight..")
                non_lora_state_dict = torch.load(os.path.join(model_args.lora_model, 'non_lora_trainables.bin'))

                for key in non_lora_state_dict.keys():
                    assert key in peft_model_dict
                    peft_model_dict[key] = non_lora_state_dict[key]


            print('update model weight with lora weight down.')

            model.load_state_dict(peft_model_dict)
                
        model.print_trainable_parameters()



    #根据 `model_args.version` 调整tokenizer中的填充token的处理方式，填充token可能被设置为特殊的 `[PAD]` token、`unk_token` 或由 version 指明的对话模板来决定。
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if data_args.prompt_version is not None:
        conversation_lib.default_conversation = conversation_lib.conv_templates[data_args.prompt_version]
    breakpoint()
    #如果存在 `vision_tower` 并且表示需要进行多模态训练，将会用各种视觉处理设置配置模型，并初始化视觉相关组件。
    if model_args.vision_tower is not None:
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
        if model_args.tune_mm_mlp_adapter:
            # model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.lora_enable:
            for k,v in model.named_parameters():
                if 'lora' in k:
                    v.requires_grad=True

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    
    #根据tokenizer和其他数据参数创建了一个数据模块用于数据预处理和加载到Trainer中
    '''数据集构建'''
    breakpoint()
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                            data_args=data_args)
    breakpoint()
    callback_class =  LLMCallback
    
    trainer = ValleyTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    callbacks=[callback_class],
                    compute_metrics= lambda x:x,
                    **data_module, 
                    )

    print_trainable_params(model)
    
    

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        # Lora model is not support this resume branch, make sure your lora out_dir is empty.
        print('resume')
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train() #开始训练
    
    trainer.save_state()

    model.config.use_cache = True

    #训练后，保存模型状态，包括具体的LoRA和非LoRA参数组的保存策略
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            tokenizer.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str,
                        default="./valley/configs/experiment_cn/valley_cn_7b_product_cp_hzh_v1_2.yaml")
    args = parser.parse_args()
    train(args)