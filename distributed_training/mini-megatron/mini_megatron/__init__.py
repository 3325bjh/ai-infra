# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Mini Megatron learning implementation for Qwen3.6-style training."""

from mini_megatron.config import (
    ConfigContainer,
    DataConfig,
    DistributedConfig,
    Qwen36ModelConfig,
    TrainConfig,
)
from mini_megatron.recipes import load_recipe, qwen36_35b_a3b_debug_config, qwen36_35b_a3b_tiny_config
from mini_megatron.training_core import TrainingSetup, TrainStepResult

__all__ = [
    "ConfigContainer",
    "DataConfig",
    "DistributedConfig",
    "Qwen36ModelConfig",
    "TrainingSetup",
    "TrainStepResult",
    "TrainConfig",
    "load_recipe",
    "qwen36_35b_a3b_debug_config",
    "qwen36_35b_a3b_tiny_config",
]
