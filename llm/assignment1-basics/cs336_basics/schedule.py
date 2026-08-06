from collections.abc import Iterable

import torch
import math
def get_lr_cosine_schedule(
        it:int,
        max_learning_rate:float,
        min_learning_rate:float,
        warmup_iters:int,
        cosine_cycle_iters:int
)->float:
    if it<warmup_iters:
        return max_learning_rate*it/warmup_iters
    if it >cosine_cycle_iters:
        return min_learning_rate
    decay_ratio=(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)
    coeff=0.5*(1+math.cos(math.pi*decay_ratio))
    return min_learning_rate+coeff*(max_learning_rate-min_learning_rate)

