import argparse
import os
import torch
import numpy as np
import wandb
from nn import TransformerLM
from optimizer import AdamW,clip_gradient_norm
from schedule import get_lr_cosine_schedule
from dataloader import get_batch
from checkpoint import save_checkpoint,load_checkpoint
from loss import cross_entropy

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--context_length",type=int,default=256)
    parser.add_argument("--d_model",type=int,default=512)
    parser.add_argument("--num_layers",type=int,default=4)
    parser.add_argument("--num_heads",type=int,default=8)
    parser.add_argument("--d_ff",type=int,default=2048)
    parser.add_argument("--vocab_size",type=int,default=10000)
    parser.add_argument("--no_rms_norm",action="store_true")
    parser.add_argument("--norm_mode",type=str,default="pre",choices=["pre","post"])
    parser.add_argument("--ffn_type",type=str,default="swiglu",choices=["swiglu","silu"])
    parser.add_argument("--lr",type=float,default=6e-4)
    parser.add_argument("--max_iters",type=int,default=10000)
    parser.add_argument("--warmup_iters",type=int,default=1000)
    parser.add_argument("--min_lr",type=float,default=6e-5)
    parser.add_argument("--max_norm",type=float,default=1.0)
    parser.add_argument("--train_data_path",type=str,required=True)
    parser.add_argument("--valid_data_path",type=str,required=True)
    parser.add_argument("--out_dir",type=str,default="out")
    parser.add_argument("--device",type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run_name",type=str,default=None,help="实验名称")
    args=parser.parse_args()
    os.makedirs(args.out_dir,exist_ok=True)
    if not os.path.exists(args.train_data_path):
        raise FileNotFoundError(f"Training data not found at{args.train_data_path}")
    if not os.path.exists(args.valid_data_path):
        raise FileNotFoundError(f"Validation data not found at{args.valid_data_path}")
    train_data=np.memmap(args.train_data_path,dtype=np.uint16,mode="r")
    val_data=np.memmap(args.valid_data_path,dtype=np.uint16,mode="r")
    print(f"训练集大小:{len(train_data)} tokens")
    print(f"验证集大小:{len(val_data)} tokens")
    actual_rope_theta=None if args.no_rope else 10000.0
    use_rm_norm=not args.no_rms_norm

    model=TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=actual_rope_theta,
        device=args.device,
        use_rms_norm=use_rm_norm,
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type
    ).to(args.devcie)

    optimizer=AdamW(model.parameters(),lr=args.lr,weight_decay=0.1)
    start_iter=0
    ckpt_path=os.path.join(args.out_dir,"ckpt.pt")
    if os.path.exists(ckpt_path):
        start_iter=load_checkpoint(ckpt_path,model,optimizer)
        print(f"Resuming from iteration {start_iter}")
    wandb.init(
        project="cs336-assignment1",
        name=args.run_name,
        config=args
    )
    for it in range(start_iter,args.max_iters):
        lr=get_lr_cosine_schedule(it,args.lr,args.min_lr,args.warmup_iters,args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr']=lr
        model.train()
        x,y=get_batch(train_data,args.batch_size,args.context_length,args.device)
        logits=model(x)
        loss=cross_entropy(logits,y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient_norm(model.parameters(),args.max_norm)
        optimizer.step()
        if it%100==0 or it==args.max_iters-1:
            model.eval()
            with torch.no_grad():
                vx,vy=get_batch(val_data,args.batch_size,args.context_length,args.device)
                v_logits=model(vx)
                v_loss=cross_entropy(v_logits,vy)
                wandb.log({
                    "train/loss":loss.item(),
                    "val/loss":v_loss.item(),
                    "lr":lr,
                    "iter":it+1
                }
                )
        if it%1000==0 and it>0:
            save_checkpoint(model,optimizer,it,ckpt_path)

    save_checkpoint(model,optimizer,args.max_iters,os.path.join(args.out_dir,"ckpt_final.pt"))
    wandb.finish()
