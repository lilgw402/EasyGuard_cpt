from typing import List, Optional, Dict, Union
import math
import os
import tempfile
from sklearn.covariance import log_likelihood
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.deepspeed import HfDeepSpeedConfig
from cruise import CruiseModule, CruiseCLI, CruiseConfig
from cruise.utilities.distributed import DIST_ENV
from cruise.utilities.hdfs_io import hcopy
from fashBloom.data.gpt.datamodule.zero_shot import ZeroShotGPTDatamodule
from fashBloom.utils.exp_helper import ExpHelper
from fashBloom.utils.generate import play_console, play_file, few_shot_play_file, play_file_qa, play_file_qa_batch
from fashBloom.optim import mariana_optimizer_kwargs_defaults


def get_subsequent_mask(seq):
    '''
    For masking out the subsequent info.
    seq: [bsz, seq_len]
    mask: [bsz, seq_len, seq_len]
    '''
    _, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    mask = seq.unsqueeze(-2) & subsequent_mask
    return mask

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).
    Basically works like a linear layer but the weights are transposed.
    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


# Config adapter
network_config = {
    "hidden_size": 2048,
    "n_embed": 2048, # 512,  # vocab embedding
    "n_inner": 8192,
    "n_head": 16,
    "n_layer": 24,
    "vocab_size": 145664,
    "max_position_embeddings": 2048, # 1025,
    "layer_norm_epsilon": 1.0e-5,
    "activation_function": "gelu_new",
    "resid_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "attn_pdrop": 0.1,
    "scale_attn_weights": True,  # TODO:
    "scale_attn_by_inverse_layer_idx": False,  # TODO:
    "reorder_and_upcast_attn": False,  # TODO:
    "initializer_range": 0.02,
    "gradient_checkpointing": False,
    "tie_weight": True,
    "pad_idx": 2,
    "use_ft_flash_attn": False,
    "use_ft_linear": False,
    "use_ft_layernorm": False,
    "use_rmpad_lmloss": False,
    "use_rmpad_lnmlp": False,
    "use_rmpad_attn": False,
  }


class GPT2Model(CruiseModule):
    """Deberta pretrain"""
    def __init__(self,
                 network: CruiseConfig = network_config,
                 freeze_prefix: Optional[List[str]] = None,
                 partial_pretrain: Optional[str] = None,
                 partial_pretrain_rename: Optional[Dict[str, str]] = None,
                 use_hf_ckpt: Optional[bool] = False,
                 model_config: Optional[str] = None,
                 ):
        super().__init__()
        self.save_hparams()  # save to self.hparams
        #TODO: check if this is needed
        self.best_val_performance_value = 0.0

        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.hparams.network.pad_idx, reduction='none')

        # if not self.hparams.use_hf_ckpt:
        #     # cruise model
        #     self.gpt = GPT2LMHeadModel(self.hparams)
        #     self.init_weights()
        #     self.freeze_params(self.hparams.freeze_prefix or [])
        if True:
            if self.hparams.partial_pretrain.startswith('hdfs'):
                tmp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(self.hparams.partial_pretrain))
                self.local_dir = tmp_dir
            else:
                self.local_dir = self.hparams.partial_pretrain
            
            print(f'local_dir: {self.local_dir}')
            if self.hparams.model_config.startswith('hdfs'):
                self.local_config = self.local_dir + "_config.json"
            else:
                self.local_config = self.hparams.model_config
            print(f'local_config: {self.local_config}')
    
    def local_rank_zero_prepare(self) -> None:
        if self.hparams.use_hf_ckpt:
            if self.hparams.partial_pretrain.startswith('hdfs'):
                hcopy(self.hparams.partial_pretrain, self.local_dir)
            if self.hparams.model_config.startswith('hdfs'):
                hcopy(self.hparams.model_config, self.local_config)

    def setup(self):
        if self.hparams.use_hf_ckpt:
            self.hf_config = AutoConfig.from_pretrained(self.local_config)
            print(f'self.local_config: {self.local_config}')
            print(f'self.hf_config: {self.hf_config}')
            self.hf_config.gradient_checkpointing = True
            self.hf_config.use_cache = False
            
            if not self.hparams.partial_pretrain:
                self.gpt = AutoModelForCausalLM.from_config(config=self.hf_config)
                self.freeze_params(self.hparams.freeze_prefix or [])
        else:
            self.hf_config = AutoConfig.from_pretrained(self.local_config)
            print(f'self.local_config: {self.local_config}')
            print(f'self.hf_config: {self.hf_config}')
            self.hf_config.gradient_checkpointing = True
            self.hf_config.use_cache = False
            
            self.gpt = AutoModelForCausalLM.from_config(config=self.hf_config)
            self.gpt.lm_head.weight=self.gpt.transformer.word_embeddings.weight
            # self._tie_or_clone_weights(self.gpt.lm_head.weight, self.gpt.transformer.word_embeddings.weight)
            # self.gpt.lm_head.weight = nn.Parameter(self.gpt.transformer.word_embeddings.weight.clone())

            self.freeze_params(self.hparams.freeze_prefix or [])
        
        self.rank_zero_print('xxxxx starting loading checkpoint')
        # In DDP rank 0 load pretrain weights is enough
        # if self.trainer.global_rank == 0 and self.hparams.partial_pretrain:
        if self.hparams.partial_pretrain:
            rename_params = self.hparams.partial_pretrain_rename or {}
            if self.hparams.use_hf_ckpt:
                self.rank_zero_print(f'load from load_from_hf')
                self.gpt = AutoModelForCausalLM.from_pretrained(
                    self.local_dir,
                    config=self.hf_config)
                
                self.rank_zero_print('MODEL ARCH: ')
                self.rank_zero_print(self.gpt)

                self.freeze_params(self.hparams.freeze_prefix or [])
            elif 'mp_rank' in self.hparams.partial_pretrain:
                # zero2 checkpoints has key 'module'
                from cruise.utilities.cloud_io import load as crs_load
                state_dict = crs_load(self.hparams.partial_pretrain, map_location='cpu')['module']
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                self.partial_load_from_checkpoints(state_dict, rename_params=rename_params)
                self.gpt.lm_head.weight=self.gpt.transformer.word_embeddings.weight
            else:
                # zero3
                self.rank_zero_print('******** zero3... ********')

                self.partial_load_from_checkpoints(
                    self.hparams.partial_pretrain,
                    rename_params=rename_params, verbose=True)
                
                self.gpt.lm_head.weight=self.gpt.transformer.word_embeddings.weight
                # self.gpt.lm_head.weight = nn.Parameter(self.gpt.transformer.word_embeddings.weight.clone())

        self.rank_zero_print('xxxxx finished loading checkpoint')

    def freeze_params(self, freeze_prefix):
        for name, param in self.named_parameters():
            for prefix in freeze_prefix:
                if name.startswith(prefix):
                    self.rank_zero_print('freeze_params:', name)
                    param.requires_grad = False

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        dataset_name=None,
        task_name=None,
        input_lens=None,
        target_tokens_list=None,
        target_idx=None,
        answer_choice_tokens_list=None,
    ):
        if not self.hparams.use_hf_ckpt:
            #print(f'before_attention_mask: {attention_mask.size()}')
            attention_mask = get_subsequent_mask(attention_mask)
            #print(f'after_attention_mask: {attention_mask.size()}')
        
        # TODO: log, remove later
        # print("print: attention_mask={}, \n input_ids={}, \n labels={} \n".format(attention_mask, input_ids, labels))
        # print(attention_mask.shape, input_ids.shape)
        # print(attention_mask[0, :100])
        # print(input_ids[0, :100])

        model_out = self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # print("output=", model_out)
        # print(model_out['logits'].shape)
        
        return model_out

    def training_step(self, batch, batch_idx):
        # log lr
        # (TODO) Need to confirm if we still need lr scheduler
        scheduler = self.trainer.lr_scheduler_configs[0].scheduler
        if hasattr(scheduler, 'get_lr'):
            self.log('lr', scheduler.get_lr()[0], console=True)
        else:
            self.log('lr', scheduler.get_last_lr()[0], console=True)
        # in hf model, labels will be shifted by 1, so here labels = input_ids
        batch['labels'] = batch['input_ids']
        #TODO: remove debug log below
        # print("batch information: {}, with batch_idx: {}".format(batch, batch_idx))
        # self.rank_zero_print("batch information: {}, with batch_idx: {}".format(batch, batch_idx))

        model_out = self.forward(**batch)
        loss = model_out['loss']

        # classification_logits = model_out['classification_logits']
        # preds = torch.argmax(classification_logits, dim=-1).long()

        # acc_nums = (torch.squeeze(batch['classification_labels']) == preds).sum().float()

        # batch_size = batch['classification_labels'].numel() * 1.0
        
        # acc = acc_nums / batch_size
        
        output = {
            "loss": loss,
        }
        # if batch_idx % 100 == 0:
        #     print("Training: loss: {}, batch_idx: {}".format(
        #                     loss, batch_idx))
        return output

    def ctx_ppl_validation(self, batch):
        batch_size, answer_choice_num, _ = batch['input_ids'].size()
        dataset_name = batch['dataset_name'][0]
        target_idx = batch['target_idx'].cpu().numpy().astype(int)
        # print(f'target_idx: {target_idx}')
        # print(f'batch_size: {batch_size}')
        # print(f'answer_choice_num: {answer_choice_num}')

        batch_ppl = []
        for answer_index in range(answer_choice_num):
            input_ids = batch['input_ids'][:,answer_index,:]
            labels = input_ids
            attention_mask = batch['attention_mask'][:,answer_index,:]
            model_out = self.forward(input_ids, attention_mask, labels)

            lm_logits = model_out['logits']
            # print(f'input_id_size: {input_ids.size()}, attention_mask_size: {attention_mask.size()}, lm_logits_size: {lm_logits.size()}')
            shift_logits = lm_logits[..., :-1, :].contiguous().permute(0,2,1)
            shift_labels = labels[..., 1:].contiguous()
            # print(f'shift_logits_size: {shift_logits.size()}')
            # print(f'shift_labels_size: {shift_labels.size()}')

            # logits: [bz x vocab x max_seq_len], labels: [bz x max_seq_len]
            # nlls: [bz x max_seq_len]
            nlls = self.loss_fct(shift_logits, shift_labels)
            answer_nlls = nlls.sum(dim=1)
            answer_counts = attention_mask.sum(dim=1)
            # print(f'nlls_size: {nlls.size()}')
            # print(f'answer_nlls: {answer_nlls.size()}')
            # print(f'answer_counts: {answer_counts.size()}')

            answer_ppl = torch.exp(answer_nlls / answer_counts).float().cpu().numpy()
            # print(f'answer_ppl: {answer_ppl}')

            batch_ppl.append(answer_ppl)
        
        pred = np.argmin(batch_ppl, 0)
        # print(f'batch_ppl: {batch_ppl}')
        # print(f'pred: {pred}')
        num_correct = sum(pred == target_idx)
        # print(f'num_sample: {batch_size}, num_correct: {num_correct}')
        
        return {'num_sample': batch_size, 'num_correct': num_correct, 'dataset_name': dataset_name}

    def multi_choice_validation(self, batch):
        batch_size, answer_choice_num, _ = batch['input_ids'].size()
        dataset_name = batch['dataset_name'][0]
        target_idx = batch['target_idx'].cpu().numpy().astype(int)
        # print(f'batch_size: {batch_size}')
        # print(f'answer_choice_num: {answer_choice_num}')

        logits_probs = []
        for answer_index in range(answer_choice_num):
            input_ids = batch['input_ids'][:,answer_index,:]
            attention_mask = batch['attention_mask'][:,answer_index,:]
            input_lens = [batch['input_lens'][i][answer_index] for i in range(batch_size)]
            answer_choice_tokens_list = [batch['answer_choice_tokens_list'][i][answer_index] for i in range(batch_size)]
            model_out = self.forward(input_ids, attention_mask)

            # bz x max_seq_len x vocab_size
            multi_logits = torch.nn.functional.log_softmax(model_out['logits'], dim=-1)
            # print(f'multi_logits: {multi_logits.size()}')
            
            logits_single_probs = []
            for input, input_len, cont_tokens, logits in zip(input_ids, input_lens, answer_choice_tokens_list, multi_logits):
                # Slice to original seq length
                cont_len = len(cont_tokens)
                # print(f'logits_size: {logits.size()}')
                # print(f'cont_token: {cont_tokens}')
                # [1, seq, vocab]
                logits = logits[input_len - cont_len : input_len, :].unsqueeze(0)
                # print(f'input_len: {input_len}, cont_len: {cont_len}')
                # print(f'logits_size: {logits.size()}')
                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                # print(f'greedy_tokens: {greedy_tokens}')
                # [1, seq]
                cont_tokens = torch.as_tensor(cont_tokens, dtype=torch.long).cuda().unsqueeze(0)
                max_equal = (greedy_tokens == cont_tokens).all()

                # Obtain logprobs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                # [1, seq]
                # print(f'before_logits: {logits.size()}')
                logits = torch.gather(logits, 2, cont_tokens.unsqueeze(-1)).squeeze(-1)
                # print(f'cont_tokens: {cont_tokens}')
                # print(f'fixed_logits: {logits}')
                # Answer: (log prob, is-exact-match)
                # answer = (float(logits.sum()), bool(max_equal))
                logits_single_probs.append(float(logits.sum()))
            logits_probs.append(logits_single_probs)
        
        # print(logits_probs)
        pred = np.argmax(logits_probs, 0)

        # print(f'compare: {pred == target_idx}')

        num_correct = sum(pred == target_idx)
        # print(f'pred: {pred}, target_idx: {target_idx}')
        # print(f'num_correct: {num_correct}')
        
        return {'num_sample': batch_size, 'num_correct': num_correct, 'dataset_name': dataset_name}
    
    def next_word_validation(self, batch):
        # print(f'batch: {batch}')
        batch_size, _ = batch['input_ids'].size()
        dataset_name = batch['dataset_name'][0]

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        input_lens = [batch['input_lens'][i] for i in range(batch_size)]
        target_tokens_list = [batch['target_tokens_list'][i] for i in range(batch_size)]
        model_out = self.forward(input_ids, attention_mask)

        #print(f'input_ids: {input_ids.size()}')
        #print(f'attention_mask: {attention_mask.size()}')
        # print(f'model_out: {model_out["logits"]}')

        # bz x max_seq_len x vocab_size
        multi_logits = torch.nn.functional.log_softmax(model_out['logits'], dim=-1)
        #print('multi_logits: ', multi_logits.size())

        num_correct = 0
        for input_len, cont_tokens, logits in zip(input_lens, target_tokens_list, multi_logits):
            cont_len = len(cont_tokens)
            #print(f'input_len: {input_len}, cont_len: {cont_len}')

            # [1, seq, vocab]
            logits = logits[input_len - cont_len : input_len, :].unsqueeze(0)
            greedy_tokens = logits.argmax(dim=-1)
            #print(f'cont_tokens: {cont_tokens}, greedy_tokens: {greedy_tokens}')

            # [1, seq]
            cont_tokens = torch.as_tensor(cont_tokens, dtype=torch.long).cuda().unsqueeze(0)
            max_equal = (greedy_tokens == cont_tokens).all().cpu().numpy()
            # print('greedy_tokens: ', greedy_tokens)
            # print('cont_tokens: ', cont_tokens)
            # print('max_equal: ', max_equal)

            # total correct num
            num_correct += max_equal
        
        # print(f'num_sample: {batch_size}, num_correct: {num_correct}')

        return {'num_sample': batch_size, 'num_correct': num_correct, 'dataset_name': dataset_name}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        task_name = batch['task_name'][0]
        if task_name == 'multi_choice_task':
            return self.multi_choice_validation(batch)
        elif task_name == 'next_word_prediction':
            return self.next_word_validation(batch)
        elif task_name == 'ctx_ppl_task':
            return self.ctx_ppl_validation(batch)
        else:
            return None
    
    def validation_epoch_end(self, outputs) -> None:
        rank_total_sample = torch.as_tensor([out['num_sample'] for out in outputs])
        # all_rank_total_sample = self.all_gather_object(rank_total_sample, sync_grads=False).reshape(-1).cpu().numpy()
        # all_rank_total_sample_list = self.all_gather_object(rank_total_sample)
        all_rank_total_sample_list = self.all_gather(rank_total_sample, sync_grads=False)
        all_rank_total_sample = [sum(num) for num in all_rank_total_sample_list]
        # print("all_rank_total_sample: {}".format(all_rank_total_sample))
        # .reshape(-1).cpu().numpy()

        # print(f'all_rank_total_sample: {all_rank_total_sample}')
        
        # all_rank_total_sample = rank_total_sample.cpu().numpy()
        # print(f'all_rank_total_sample: {all_rank_total_sample}')

        rank_total_correct = torch.as_tensor([out['num_correct'] for out in outputs])
        # all_rank_total_correct = self.all_gather_object(rank_total_correct, sync_grads=False).reshape(-1).cpu().numpy()
        # all_rank_total_correct_list = self.all_gather_object(rank_total_correct)#.reshape(-1).cpu().numpy()
        all_rank_total_correct_list = self.all_gather(rank_total_correct, sync_grads=False)
        all_rank_total_correct = [sum(num) for num in all_rank_total_correct_list]
        # print("all_rank_total_correct: {}".format(all_rank_total_correct))
        # print(f'all_rank_total_correct: {all_rank_total_correct}')
        # print(f'rank_total_correct: {rank_total_correct.size()}')
        # all_rank_total_correct = rank_total_correct.cpu().numpy()
        # print(f'all_rank_total_correct: {all_rank_total_correct}')

        dataset_name = outputs[0]['dataset_name']

        acc = sum(all_rank_total_correct) * 1.0 / sum(all_rank_total_sample)

        if acc >= self.best_val_performance_value:
            self.best_val_performance_value = acc

        self.rank_zero_info(f"Zero Shot Learning Acc: all_rank_total_sample:{sum(all_rank_total_sample)}, all_rank_total_correct:{sum(all_rank_total_correct)}, val_acc_per_epoch of current step: {acc}")
        self.rank_zero_info(f"Zero Shot Learning Acc: best val_acc_per_epoch so far: {self.best_val_performance_value}")

        self.log_dict({
            f'{dataset_name}/acc': acc,
            f'{dataset_name}/num_sample': sum(all_rank_total_sample),
            f'{dataset_name}/num_correct': sum(all_rank_total_correct)
        }, console=True)

        return acc

    @torch.no_grad()
    def decode(self, input_ids: torch.Tensor, input_mask: torch.Tensor, *args, **kwargs):
        """For generation task"""
        # TODO: remove log later
        # print("print: decode being called")

        model_out = self.gpt(input_ids=input_ids, attention_mask=input_mask)
        return model_out

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hparams.network.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if "c_proj.weight" in name or 'attn_ow' in name: # deepspeed transformer kernel 是 attn_ow
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.hparams.network.initializer_range / math.sqrt(2 * self.hparams.network.n_layer)))


if __name__ == '__main__':
    helper = ExpHelper(__file__)
    from cruise.trainer.callback import ModelCheckpoint
    ckpter = ModelCheckpoint(monitor='step',
                             save_last=False,
                             save_top_k=-1,
                             every_n_train_steps=10000,
                             every_n_epochs=1,
                             save_on_train_epoch_end=True,
                             enable_trace=False)
    cli = CruiseCLI(
        GPT2Model,
        datamodule_class=ZeroShotGPTDatamodule,
        trainer_defaults={
            'precision': 16,
            'enable_versions': False,
            'find_unused_parameters': False,
            'max_epochs': 1,
            "default_hdfs_dir": helper.hdfs_prefix,
            "project_name": helper.project_name,
            'val_check_interval': -1,
            'summarize_model_depth': 2,
            'gradient_clip_val': 1.0,
            'checkpoint_monitor': 'step',
            'checkpoint_mode': 'max',
            'callbacks': [ckpter],
            'optimizer_kwargs': mariana_optimizer_kwargs_defaults,
            })
    cli.add_argument('--val-only', default=False, action='store_true', dest='val_only')
    cli.add_argument('--play', default=False, action='store_true', dest='play')
    cli.add_argument('--play-file', default='', type=str, help='generate by samples loaded from file')
    cli.add_argument('--play-out-file', default='', type=str, help='generate by samples loaded from file', dest='play_out_file')
    cli.add_argument('--dataset-name', default='', type=str, help='dataset name', dest='dataset_name')
    cli.add_argument('--subset-name', default='', type=str, help='subset name', dest='subset_name')
    cli.add_argument('--template-name', default='', type=str, help='template name', dest='template_name')
    cli.add_argument('--play-file-type', default='none', type=str, help='generate by samples loaded from file for qa')
    cli.add_argument('--play-file-limit', default=-1, type=int, help="If >0, limit how many lines to generate.")
    cli.add_argument('--generate-trial-num', default=1, type=int, help="generation trial num, default is 5")
    cli.add_argument('--generate-steps', default=256, type=int, help='decode sequence length/steps')
    cli.add_argument('--generate-temp', default=0.7, type=float, help='Smaller tempreature logits become more steep')
    cli.add_argument('--generate-do-sample', default=True, type=bool, help='multinomial sample if True')
    cli.add_argument('--generate-topk', default=5, type=int, help='sample top-k')
    cli.add_argument('--generate-topp', default=None, type=float, help='sample at least top-p probability')
    cli.add_argument('--generate-dynamic-topp', default=None, type=float, help='sample at least dynamic top-p probability')
    cli.add_argument('--generate-dynamic-topp-omega', default=0.3, type=float, help='omega for dynamic topp')
    cli.add_argument('--generate-dynamic-topp-decay', default=0.9, type=float, help='lambda for dynamic toppp') 
    cli.add_argument('--generate-n-eos', default=1, type=int, help='Stop until n-eos tokens')
    cli.add_argument('--num-fewshot', default=0, type=int, help='fewshot to control')
    cli.add_argument('--fewshot-file-path', default='', type=str, help='few shot file path')


    cfg, trainer, model, datamodule = cli.parse_args()

    try:
        hfds_config_file_or_dict = cfg['trainer']['accelerator_kwargs']['ds_config']
        hfds_config = HfDeepSpeedConfig(hfds_config_file_or_dict)
    except:
        pass
    
    if cfg.val_only:
        trainer.validate(model, datamodule=datamodule)
    elif cfg.play_file or cfg.play:
        # assert DIST_ENV.world_size == 1, "Play mode only support single card"
        datamodule.rank_zero_prepare()
        datamodule.local_rank_zero_prepare()
        datamodule.setup()
        tokenizer = datamodule.tokenizer
        assert tokenizer is not None, "Invalid tokenizer from datamodule"
        model.rank_zero_prepare()
        model.local_rank_zero_prepare()
        model.setup()

        if cfg.play_file:
            print("\nFile play mode.")
            if cfg.play_file_type == "qa_batch":
                play_file_qa_batch(cfg.play_file, cfg.play_out_file, datamodule, tokenizer, model.cuda(), cfg.generate_trial_num,
                      steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                      top_k=cfg.generate_topk, top_p=cfg.generate_topp,
                      until_n_eos=cfg.generate_n_eos,
                      limit_samples=cfg.play_file_limit)                
            elif cfg.play_file_type == "qa":
                play_file_qa(cfg.play_file, cfg.play_out_file, tokenizer, model.cuda(), cfg.generate_trial_num,
                      steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                      top_k=cfg.generate_topk, top_p=cfg.generate_topp,
                      until_n_eos=cfg.generate_n_eos,
                      limit_samples=cfg.play_file_limit)
            else:
                few_shot_play_file(cfg.play_file, cfg.play_out_file, tokenizer, model.cuda(), cfg.generate_trial_num,
                        steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                        top_k=cfg.generate_topk, top_p=cfg.generate_topp, until_n_eos=cfg.generate_n_eos,
                        limit_samples=cfg.play_file_limit, dataset_name=cfg.dataset_name, subset_name=cfg.subset_name, template_name=cfg.template_name,
                        num_fewshot=cfg.num_fewshot,
                        fewshot_file_path=cfg.fewshot_file_path)
        else:
            print("\nConsole play mode.")
            play_console(tokenizer, model.cuda(), cfg.generate_trial_num,
                         steps=cfg.generate_steps, temperature=cfg.generate_temp, do_sample=cfg.generate_do_sample,
                         top_k=cfg.generate_topk, top_p=cfg.generate_topp,
                         until_n_eos=cfg.generate_n_eos)
    else:
        trainer.fit(model, datamodule)

