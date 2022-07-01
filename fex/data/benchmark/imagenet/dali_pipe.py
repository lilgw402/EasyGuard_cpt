# -*- coding: utf-8 -*-
'''
imagenet dali preprocess pipeline
'''
import random
import os
import numpy as np
import torch
import multiprocessing.dummy as mp

try:
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
    import nvidia.dali.fn as fn
except ImportError:
    Pipeline = object
    print("DALI is not installd, please install it from https://www.github.com/NVIDIA/DALI")


def ExternalSourcePipeline(external_data, batch_size, size=256, crop=224,
                           is_training=False,
                           dali_cpu=False, num_threads=2, device_id=0):
    pipe = Pipeline(batch_size, num_threads, device_id)
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'

    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        if is_training:
            # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
            device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
            host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
            # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
            preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
            preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
            images = fn.decoders.image_random_crop(jpegs,
                                                   device=decoder_device, output_type=types.RGB,
                                                   device_memory_padding=device_memory_padding,
                                                   host_memory_padding=host_memory_padding,
                                                   preallocate_width_hint=preallocate_width_hint,
                                                   preallocate_height_hint=preallocate_height_hint,
                                                   random_aspect_ratio=[0.8, 1.25],
                                                   random_area=[0.1, 1.0],
                                                   num_attempts=100)
            images = fn.resize(images,
                               device=dali_device,
                               resize_x=crop,
                               resize_y=crop,
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)
        else:
            images = fn.decoders.image(jpegs,
                                       device=decoder_device,
                                       output_type=types.RGB)
            images = fn.resize(images,
                               device=dali_device,
                               size=size,
                               mode="not_smaller",
                               interp_type=types.INTERP_TRIANGULAR)
            mirror = False
        images = fn.crop_mirror_normalize(images.gpu(),
                                          dtype=types.FLOAT,
                                          output_layout="CHW",
                                          crop=(crop, crop),
                                          mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                          std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                                          mirror=mirror)
        labels = labels.gpu()
        pipe.set_outputs(images, labels)
    return pipe
