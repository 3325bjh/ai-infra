import torch
import json
import random
import wandb
import os
import argparse
import numpy as np
from sympy.physics.units import temperature
from tqdm import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM,AutoTokenizer
from vllm import LLM,SamplingParams
from unittest.mock import patch
from typing import Dict

from cs336_alignment.sft_utils import (
    tokenize_prompt_and_output,
    sft_microbatch_train_step,
    log_generations,
    get_response_log_probs
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, grade
from tests.conftest import reward_fn, policy_log_probs


def get_batch(
    tokenized_data: Dict[str, torch.Tensor],
    batch_size: int,
    device: str | torch.device,
) -> Dict[str, torch.Tensor]:
    """Sample a training batch from pre-tokenized examples.

    Sampling is done with replacement so this function also works when a
    micro-batch is larger than the dataset.  The index tensor stays on the
    source device (normally CPU), while the returned batch is moved to the
    requested training device.
    """
    if not tokenized_data:
        raise ValueError("tokenized_data must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    first_tensor = next(iter(tokenized_data.values()))
    if not isinstance(first_tensor, torch.Tensor) or first_tensor.ndim == 0:
        raise ValueError("tokenized_data values must be batched tensors")
    num_examples = first_tensor.shape[0]
    if num_examples == 0:
        raise ValueError("tokenized_data must contain at least one example")

    for name, tensor in tokenized_data.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            raise ValueError(f"{name} must be a batched tensor")
        if tensor.shape[0] != num_examples:
            raise ValueError("all tokenized tensors must have the same number of examples")

    indices = torch.randint(num_examples, (batch_size,))
    return {
        name: tensor.index_select(0, indices).to(device)
        for name, tensor in tokenized_data.items()
    }


def init_vllm(model_id,device,seed,gpu_memory_utilization):
    with patch("torch.distributed.get_world_size",return_value=1),\
        patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",return_value=None):
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed
        )

def load_policy_into_vllm_instance(policy,llm):
    state_dict=policy.state_dict()
    llm_model=llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())
    print("\n[Sync] Policy weights synced to vLLM")

def run_sft_experiment(args):
    grad_accum_steps=args.gradient_accumulation_steps
    wandb.init(project=args.wandb_project,name=args.wandb_run_name,config=vars(args))
    with open(args.prompt_path,"r") as f:
        rl_template=f.read().strip()
    print(f"Initializing Model:{args.model_id}")
    tokenizer=AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    policy=AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"
    ).to(args.device)
    policy.gradient_checkpointing_enable()
    optimizer=AdamW(policy.parameters(),lr=args.lr)
    print(f"Initializing vLLM on {args.vllm_device}...")
    vllm_inst=init_vllm(args.model_id,args.vllm_device,args.seed,args.vllm_gpu_util)
    print(f"Loading training data from {args.train_data_path}...")
    raw_train_data=[]
    with open(args.train_data_path,"r") as f:
        for line in f:
            raw_train_data.append(json.loads(line))
    if args.filter_correct:
        print("filtering correct examples...")
        raw_train_data=[item for item in raw_train_data if item.get("is_correct",True)]
        print(f"filtered data size:{len(raw_train_data)}")
    print("Pre-tokenizing entire training dataset...")
    tokenized_train_data=tokenize_prompt_and_output(
        prompt_strs=[item["prompt"] for item in raw_train_data],
        output_strs=[item["response"] for item in raw_train_data],
        tokenizer=tokenizer
    )
    print(f"Tokenization complete.Total samples:{len(tokenized_train_data['input_ids'])}")
    print(f"loading validation data from {args.val_data_path}...")
    val_prompts=[]
    val_ground_truths=[]
    with open(args.val_data_path,"r") as f:
        for i,line in enumerate(f):
            if i>=args.max_eval_samples:break
            item=json.loads(line)
            raw_a=item["answer"]
            gold=raw_a.split("####")[-1].strip() if "####" in raw_a else raw_a.strip()
            formatted_prompt=rl_template.replace("{question}",item["question"])
            val_prompts.append(formatted_prompt)
            val_ground_truths.append(gold)
    eval_sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    progress_bar=tqdm(range(args.max_steps),desc="SFT Steps")
    print(f"\n[Step 0] Starting Evaluation...")
    policy.eval()
    load_policy_into_vllm_instance(policy,vllm_inst)
    metrics=log_generations(
        vllm_model=vllm_inst,
        sampling_params=eval_sampling_params,
        prompts=val_prompts,
        ground_truths=val_ground_truths,
        reward_fn=r1_zero_reward_fn,
        step=0,
        log_prefix="eval"
    )
    print(f"Eval Accuracy:{metrics.get('eval/accuracy',0):.2%}")
    policy.train()
    for step in range(args.max_steps):
        accumulated_loss=0.0
        accumulated_entropy=0.0
        accumulated_res_entropy=0.0
        for _ in range(grad_accum_steps):
            batch=get_batch(tokenized_train_data,args.micro_batch_size,args.device)
            response_outputs=get_response_log_probs(
                model=policy,
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                return_token_entropy=True
            )
            log_probs=response_outputs["log_probs"]
            token_entropy=response_outputs["token_entropy"]
            with torch.no_grad():
                valid_token_mask=(batch["labels"]!=tokenizer.pad_token_id)
                current_res_mask=batch["response_mask"].bool()&valid_token_mask
                avg_res_entropy=token_entropy[current_res_mask].mean().item() if current_res_mask.any() else 0.0
                avg_global_entropy=token_entropy[valid_token_mask].mean().item()
            loss,_=sft_microbatch_train_step(
                policy_log_probs=log_probs,
                response_mask=batch["response_mask"],
                gradient_accumulation_steps=grad_accum_steps,
                # normalize_constant=batch["response_mask"].sum().item()
                normalize_constant = 1.0
            )
            accumulated_loss+=loss.item()*grad_accum_steps
            accumulated_entropy+=avg_global_entropy
            accumulated_res_entropy+=avg_res_entropy
        torch.nn.utils.clip_grad_norm_(policy.parameters(),1.0)
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update()
        wandb.log({
            "train/loss":accumulated_loss/grad_accum_steps,
            "train/global_entropy":accumulated_entropy/grad_accum_steps,
            "train/response_entropy":accumulated_res_entropy/grad_accum_steps,
            "train_step":step+1
        })
        if (step+1)%args.eval_every_steps==0:
            print(f"\n[Step{step+1}] Starting Evaluation...")
            policy.eval()
            load_policy_into_vllm_instance(policy,vllm_inst)
            metrics = log_generations(
                vllm_model=vllm_inst,
                sampling_params=eval_sampling_params,
                prompts=val_prompts,
                ground_truths=val_ground_truths,
                reward_fn=r1_zero_reward_fn,
                step=step+1,
                log_prefix="eval"
            )
            print(f"Eval Accuracy:{metrics.get('eval/accuracy', 0):.2f}")
            policy.train()
    print("Training finished.Saving model...")
    save_name=f"sft_steps{args.max_steps}_subset{args.dataset_size}_filtered{args.filter_correct}"
    output_dir=os.path.join(args.output_dir,save_name)
    os.makedirs(output_dir,exist_ok=True)
    policy.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    wandb.finish()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="SFT")
    parser.add_argument("--model_id",type=str,default="model/Qwen2.5-Math-1.5B")
    parser.add_argument("--train_data_path",type=str,default="data/gsm8k/train.jsonl")
    parser.add_argument("--val_data_path",type=str,default="data/gsm8k/test.jsonl")
    parser.add_argument("--prompt_path",type=str,default="cs336_alignment/prompts/r1_zero.prompt")
    parser.add_argument("--output_dir",type=str,default="result/checkpoints")

    parser.add_argument("--lr",type=float,default=2e-5)
    parser.add_argument("--batch_size",type=int,default=16)
    parser.add_argument("--micro_batch_size",type=int,default=1)
    parser.add_argument("--max_steps",type=int,default=200)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--max_tokens",type=int,default=1024)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=None)

    parser.add_argument("--dataset_size",type=int,default=None)
    parser.add_argument("--filter_correct",action="store_true")

    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--vllm_device",type=str,default="cuda:1")
    parser.add_argument("--vllm_gpu_util",type=float,default=0.5)
    parser.add_argument("--eval_every_steps",type=int,default=20)
    parser.add_argument("--max_eval_samples",type=int,default=100)

    parser.add_argument("--wandb_project",type=str,default="cs336-sft")
    parser.add_argument("--wandb_run_name",type=str,default=None)

    args=parser.parse_args()
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = max(
            1, (args.batch_size + args.micro_batch_size - 1) // args.micro_batch_size
        )
    run_sft_experiment(args)
