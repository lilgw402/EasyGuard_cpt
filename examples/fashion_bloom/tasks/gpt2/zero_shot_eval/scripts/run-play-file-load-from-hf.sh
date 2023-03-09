#!/bin/bash
set -x

declare -A model_dir
model_dir=(
    ['bloom_7b1']='hdfs://haruna/home/byte_ecom_govern/user/doushihan/hf_models/bloom-7b1'
    ['bloom_560m']='hdfs://haruna/home/byte_ecom_govern/user/doushihan/hf_models/bloom560m/bloomz-560m'
)

chkpt_path=${model_dir[$@]}

bash launch.sh tasks/gpt2/zero_shot_eval/model.py \
  --model.use_hf_ckpt=True \
  --data.from_hf_tokenizer=True \
  --data.tokenizer="$chkpt_path" \
  --data.hf_tokenizer_use_fast=False \
  --data.max_seq_len=512 \
  --model=tasks/gpt2/zero_shot_eval/1b.yaml \
  --model.partial_pretrain="$chkpt_path" \
  --model.model_config="$chkpt_path/config.json" \
  --data.train_num_workers=1 \
  --data.train_batch_size=1 \
  --data.val_num_workers=1 \
  --data.val_batch_size=1 \
  --trainer.val_check_interval=0.5 \
  --data.train_path=hdfs://haruna/home/byte_data_aml_research/user/zhengyu.chen/lambada/dev_4864/dev.parquet  \
  --data.val_path=hdfs://haruna/home/byte_data_aml_research/user/yao.cheng/lambada/clean_test \
  --data.dataset_name=lambada \
  --data.template_name=please+next+word  \
  --data.drop_last=True \
  --model.network.use_rmpad_lmloss=false \
  --model.network.use_rmpad_lnmlp=false \
  --model.network.use_rmpad_attn=false \
  --model.network.pad_idx=3 \
  --trainer.max_epochs=20 \
  --trainer.optimizer_kwargs.optimizer.params.lr=1e-5 \
  --play-file-type="qa" \
  --play-file /mnt/bn/ecom-govern-maxiangqian/doushihan/data/data/test_0130_add_trans2_prompt.jsonl \
  --play-out-file /mnt/bn/ecom-govern-maxiangqian/doushihan/play_file_out/easyguard_test.jsonl \
  --generate-temp 0.7

