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

from __future__ import annotations

from collections.abc import Mapping


def causal_lm_forward_step(batch: Mapping[str, object], model: object) -> tuple[object, dict[str, float]]:
    """Run one causal language-model forward step.

    Tutorial chapter: 5. Forward step and loss.
    Reading route:
    - `3rdparty/Megatron-LM/pretrain_gpt.py`
    - `src/megatron/bridge/training/gpt_step.py`
    """

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    # Keep the call compatible with the causal-LM model interface.  In
    # particular, use keyword arguments so this remains correct if the model
    # later grows optional inputs such as position_ids or attention_mask.
    model_output = model(input_ids=input_ids, labels=labels)  # type: ignore[operator]

    loss = getattr(model_output, "loss", None)
    if loss is None:
        raise ValueError("causal language model output must contain a loss")

    # The training core backpropagates the returned loss object, while the
    # metric dictionary must contain an ordinary Python float for logging.
    detached_loss = loss.detach() if hasattr(loss, "detach") else loss
    loss_value = detached_loss.item() if hasattr(detached_loss, "item") else detached_loss

    aux_loss = getattr(model_output, "aux_loss", None)
    metrics = {"loss": float(loss_value)}
    lm_loss = getattr(model_output, "lm_loss", None)
    if lm_loss is not None:
        detached_lm_loss = lm_loss.detach() if hasattr(lm_loss, "detach") else lm_loss
        lm_loss_value = (
            detached_lm_loss.item()
            if hasattr(detached_lm_loss, "item")
            else detached_lm_loss
        )
        metrics["lm loss"] = float(lm_loss_value)
    if aux_loss is not None:
        detached_aux_loss = aux_loss.detach() if hasattr(aux_loss, "detach") else aux_loss
        aux_loss_value = (
            detached_aux_loss.item()
            if hasattr(detached_aux_loss, "item")
            else detached_aux_loss
        )
        metrics["aux loss"] = float(aux_loss_value)

    return loss, metrics

