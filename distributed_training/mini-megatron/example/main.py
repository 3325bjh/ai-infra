import torch
import torch.distributed as dist
from example.ddp import all_reduce_grads
from example.shceduler import forward_backward_1f1b
from init_parallel import init_parallel
from layers import ParallelMLP
def main():
    init_parallel(tp_size=2,pp_size=2)
    h,seq,micro=256,(16,8,256),8
    layers_per_stage=2
    stage=torch.nn.Sequential(*[ParallelMLP(h) for _ in range(layers_per_stage)])
    all_reduce_grads_targets=stage
    opt=torch.optim.Adam(stage.parameters(),lr=1e-4)
    def data_iter_gen():
        while True: yield torch.randn(*seq)
    data_iter=data_iter_gen()
    def loss_fn(y): return y.float().pow(2).mean()
    for step in range(100):
        opt.zero_grad()
        forward_backward_1f1b(stage,data_iter,micro,seq,loss_fn)
        all_reduce_grads(all_reduce_grads_targets)
        opt.step()
        if dist.get_rank()==0 and step%10==0:
            print(f"step {step} done")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()