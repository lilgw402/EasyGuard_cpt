# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Configuration base class and utilities."""
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .. import __version__
from ..utils import logging

logger = logging.get_logger(__name__)


# TODO (junwei.Dong): 一些通用的关于config的操作可以注册进该基类中去, 自己的config最好继承该基类
# EasyGuard config base class
@dataclass
class ConfigBase(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def asdict(self) -> Dict[str, Any]:
        return asdict(self)

    def config_update_for_pretrained(self, **kwargs):
        ...
