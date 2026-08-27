import numpy as np
import wandb
from sympy.physics.units.systems.si import dimex
from vllm import LLM,SamplingParams
from transformers import PreTrainedTokenizer,PreTrainedModel
from typing import List,Dict,Callable
import torch
import torch.nn.functional as F
def tokenize_prompt_and_output(
        prompt_strs:List[str],
        output_strs:List[str],
        tokenizer:PreTrainedTokenizer
)-> Dict[str,torch.Tensor]:
    all_input_ids=[]
    all_response_masks=[]
    all_lengths=[]
    for p_str,o_str in zip(prompt_strs,output_strs):
        p_ids=tokenizer.encode(p_str)
        o_ids=tokenizer.encode(o_str)
        combined_ids=p_ids+o_ids
        all_input_ids.append(combined_ids)
        all_lengths.append(len(combined_ids))
        mask=[0]*len(p_ids)+[1]*len(o_ids)
        all_response_masks.append(mask)
    max_len=max(all_lengths)
    batch_size=len(prompt_strs)
    pad_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    padded_input_ids=torch.full([batch_size,max_len],pad_id,dtype=torch.long)
    padded_masks=torch.zeros([batch_size,max_len],dtype=torch.long)
    for i,(ids,m) in enumerate(zip(all_input_ids,all_response_masks)):
        length=len(ids)
        padded_input_ids[i,:length]=torch.tensor(ids)
        padded_masks[i,:length]=torch.tensor(m)
    final_input_ids=padded_input_ids[:,:-1]
    final_labels=padded_input_ids[:,1:].clone()
    final_response_mask=padded_masks[:,1:]

    return {
        "input_ids":final_input_ids,
        "labels":final_labels,
        "response_mask":final_response_mask
    }

def compute_entropy(logits:torch.Tensor)->torch.Tensor:
    lse=torch.logsumexp(logits,dim=-1)
    probs=F.softmax(logits,dim=-1)
    #E[z]=sum(p_i*z_i)
    exp_logits=torch.sum(probs*logits,dim=-1)
    #LSE-E[z]
    entropy=lse-exp_logits
    return entropy

def get_response_log_probs(
        model:PreTrainedModel,
        input_ids:torch.Tensor,
        labels:torch.Tensor,
        return_token_entropy:bool=False,
)->Dict[str,torch.Tensor]:
    output=model(input_ids)
    logits=output.logits #(batch_size,seq_len,vocab_size)
    log_probs_all=F.log_softmax(logits,dim=-1) #(batch_size,seq,vocab_size)
    log_probs=torch.gather(
        log_probs_all,
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)
    results={"log_probs":log_probs}
    if return_token_entropy:
        results["token_entropy"]=compute_entropy(logits)
    return results

def log_generations(
        vllm_model:LLM,
        sampling_params:SamplingParams,
        prompts:list[str],
        ground_truths:List[str],
        reward_fn:Callable[[str,str],Dict[str,float]],
        step:int,
        log_prefix:str="eval"
):
    outputs=vllm_model.generate(prompts, sampling_params)
    table_data=[]
    all_lengths=[]
    correct_lengths=[]
    incorrect_lengths=[]
    total_reward=0
    total_format_reward=0
    total_answer_reward=0
    for i,output in enumerate(outputs):
        generated_text=output.outputs[0].text
        gold_answer=ground_truths[i]
        scores=reward_fn(generated_text,gold_answer)
        r=scores.get("reward",0.0)
        fr=scores.get("format_reward",0.0)
        ar=scores.get("answer_reward",0.0)
        resp_len=len(generated_text)
        all_lengths.append(resp_len)
        if r>0.5:
            correct_lengths.append(resp_len)
        else:
            incorrect_lengths.append(resp_len)
        total_reward+=r
        total_format_reward+=fr
        total_answer_reward+=ar
        if i<100:
            table_data.append([
                step,
                prompts[i],
                generated_text,
                gold_answer,
                r,fr,ar
            ])
    metrics={
        f"{log_prefix}/accuracy":total_reward/len(prompts),
        f"{log_prefix}/format_score":total_format_reward/len(prompts),
        f"{log_prefix}/answer_score":total_answer_reward/len(prompts),
        f"{log_prefix}/avg_length":np.mean(all_lengths),
        f"{log_prefix}/avg_length_correct":np.mean(correct_lengths) if correct_lengths else 0,
        f"{log_prefix}/avg_length_incorrect":np.mean(incorrect_lengths) if incorrect_lengths else 0,
    }
    if wandb.run is not None:
        columns=["step","prompt","response","ground_truth","reward","format_reward","answer_reward"]
        wandb.log({f"{log_prefix}/samples":wandb.Table(columns=columns,data=table_data)},step=step)

    return metrics

def masked_normalize(
        tensor:torch.Tensor,
        mask:torch.Tensor,
        normalize_constant:float,
        dim:int|None=None
)->torch.Tensor:
    masked_tensor=tensor*mask
    if dim is None:
        total_sum=torch.sum(masked_tensor)
    else:
        total_sum=torch.sum(masked_tensor,dim=dim)
    return total_sum/normalize_constant


def sft_microbatch_train_step(
        policy_log_probs:torch.Tensor,
        response_mask:torch.Tensor,
        gradient_accumulation_steps:int,
        normalize_constant:float=1.0
)->tuple[torch.Tensor,dict[str,torch.Tensor]]:
    batch_size=policy_log_probs.shape[0]
    nil_per_token=-policy_log_probs
    total_masked_loss=masked_normalize(
        tensor=nil_per_token,
        mask=response_mask,
        normalize_constant=normalize_constant,
        dim=None
    )
    microbatch_loss_mean=total_masked_loss/batch_size
    scaled_loss=microbatch_loss_mean/gradient_accumulation_steps
    scaled_loss.backward()
    metadata={
        "loss":microbatch_loss_mean.detach(),
    }
    return scaled_loss,metadata
