# -*- coding: utf-8 -*-
'''
Created on Nov-25-20 15:18
imagenet_dist_dataset.py
@author: liuzhen.nlp
Description:
'''
import torch
from typing import List, Dict, Any

from cruise.data_module import CruiseDataModule
from cruise.utilities.distributed import DIST_ENV

from ..dataset.tusou_json import TusouJsonDataset
from ..dali_pipeline import TrainImageDecoderPipeline
from ..dali_iter import PytorchDaliIter
from ...benchmark_dataloader import DelegateBenchmarkLoader


class TusouDataModule(CruiseDataModule):
    """Tusou dataset module.

    The trainset creates the identical dataset from TSJsonDataset in fex, while the
    validation loader only returns one or two items, where each item is the name of a
    benchmark to run.

    Args:
        val_kwargs (dict): any kwargs to pass to the val benchmark.

    """
    def __init__(self,
                 vocab_file: str = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/modelhub/albert_6l_zh_mix_oldcut_20200921/archer/zh_old_cut_145607.vocab',
                 max_len: int = 24,
                 train_path: str = 'hdfs://haruna/home/byte_search_nlp_lq/user/huangwenguan/data/tusou/tbase_20210117_rsz_score0.5_p_cln',
                 train_size: int = 1031469047,
                 train_batch_size: int = 20,
                 train_num_workers: int = 8,
                 prefetch_queue_depth: int = 4,
                 need_text: bool = True,
                 need_temb: bool = False,
                 is_bert_style: bool = True,
                 is_clip_style: bool = False,
                 train_model_input: List[str] = ['image'],
                 val_benchmarks: List[str] = ['tusou_recall'],
                 val_kwargs: Dict[str, Any] = {},
                 ):
        super().__init__()
        self.save_hparams()

    def train_dataloader(self):
        train_dataset = TusouJsonDataset(
            self.hparams.vocab_file,
            self.hparams.max_len,
            self.hparams.train_path,
            rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            shuffle=True,
            repeat=True,
            data_size=self.hparams.train_size,
            need_temb=self.hparams.need_temb,
            is_bert_style=self.hparams.is_bert_style,
            is_clip_style=self.hparams.is_clip_style
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.train_num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=train_dataset.collect_fn)

        train_pipeline = TrainImageDecoderPipeline(
            batch_size=self.hparams.train_batch_size,
            num_threads=4,
            device_id=DIST_ENV.local_rank,
            external_data=train_loader,
            crop=224,
            prefetch_queue_depth=self.hparams.prefetch_queue_depth,
            need_text=self.hparams.need_text,
            need_temb=self.hparams.need_temb,
            random_area_min=0.4)
        train_dali_iter = PytorchDaliIter(
            dali_pipeline=train_pipeline,
            output_map=self.hparams.train_model_input,
            auto_reset=True,
            last_batch_padded=True,
            fill_last_batch=False,
            size=len(train_loader))

        return train_dali_iter

    def val_dataloader(self):
        return DelegateBenchmarkLoader(
            self.hparams.val_benchmarks,
            vocab_file=self.hparams.vocab_file,
            **self.hparams.val_kwargs)
