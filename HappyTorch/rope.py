import torch
def apply_rotary_pos_emb(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
   batch, num_heads, seq_len, head_dim=x.shape
   power=torch.arange(0,head_dim,2)/head_dim
   #(head_dim/2)
   freq=1/(10000**(torch.arange(0,head_dim,2,device=x.device).float()/head_dim))
   #(batch,seq_len,1)
   pos=pos.unsqueeze(-1).float()
   #(batch,seq_len,head_dim/2)
   angles=pos*freq.unsqueeze(0).unsqueeze(0)
   cos=torch.cos(angles)
   sin=torch.sin(angles)
   #(batch,num_heads,seq_len,head_dim/2)
   x1=x[...,0::2]
   x2=x[...,1::2]
   #(batch,1,seq_len,head_dim/2)
   cos=cos.unsqueeze(1)
   sin=sin.unsqueeze(1)

   x_rotated_1=x1*cos-x2*sin
   x_rotated_2=x1*sin+x2*cos
   x_out=torch.stack([x_rotated_1,x_rotated_2],dim=-1).flatten(-2)

   return x_out


