import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from BitNets import BitLinear


def softmax_one(x, dim=None):
    #subtract the max for stability
    x = x - x.max(dim=dim, keepdim=True).values
    #compute exponentials
    exp_x = torch.exp(x)
    #compute softmax values and add on in the denominator
    return exp_x / (1 + exp_x.sum(dim=dim, keepdim=True))

class Attn(nn.Module):
  def __init__(self, embed_dim, n_head, softmax_mode="vanilla", P_Mode="FP"):
    super(Attn, self).__init__()
    assert embed_dim % n_head == 0, f"embed_dim: {embed_dim} should be divisible by head_dim: {n_head}"
    if P_Mode == "FP":
      self.Q = nn.Linear(embed_dim, embed_dim)
      self.K = nn.Linear(embed_dim, embed_dim)
      self.V = nn.Linear(embed_dim, embed_dim)
      self.O = nn.Linear(embed_dim, embed_dim)
    elif P_Mode == "1bit":
      self.Q = BitLinear(embed_dim, embed_dim)
      self.K = BitLinear(embed_dim, embed_dim)
      self.V = BitLinear(embed_dim, embed_dim)
      self.O = BitLinear(embed_dim, embed_dim)
    self.n_head = n_head
    self.softmax_mode = softmax_mode

  def forward(self, x):
    B, T, C = x.size()
    Qs = self.Q(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
    Ks = self.K(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
    Vs = self.V(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)

    QK = (Qs @ Ks.transpose(-1,-2)) * (1/math.sqrt(Ks.size(-1)))
    causal_mask = torch.triu(torch.ones(T, T) * float('-inf'), diagonal=1).to(QK.device)
    masked_QK = QK + causal_mask[None, None, :, :]

    if self.softmax_mode == "off1":
      attn_score = softmax_one(masked_QK, dim=-1)
    else:
      attn_score = F.softmax(masked_QK, dim=-1)

    OV = self.O((attn_score @ Vs).transpose(1,2).contiguous().view(B, T, C))

    return OV, attn_score


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, seq_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_len, embed_size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class IHCModel(nn.Module):
  def __init__(self, embed_dim, n_head, vocab_size, n_layers, block_size, Position_mode="learnable", norm_mode="no_norm", softmax_mode="vanilla", P_Mode="FP"):
    super(IHCModel, self).__init__()
    self.n_layers = n_layers
    self.Position_mode = Position_mode
    self.norm_mode = norm_mode
    self.embed = nn.Embedding(vocab_size, embed_dim)

    if Position_mode == "learnable":
      self.pos_embed = nn.Parameter(torch.ones(1, block_size, embed_dim))
    elif Position_mode == "trigonometric":
      self.pos_embed = PositionalEncoding(embed_dim)

    if norm_mode != "no_norm":
      self.norm = nn.LayerNorm(embed_dim)

    self.induction_heads = nn.ModuleList([Attn(embed_dim, n_head, softmax_mode=softmax_mode, P_Mode=P_Mode) \
                                         for _ in range(n_layers)])

    self.unembed = nn.Linear(embed_dim, vocab_size, bias=False)
    self.apply(self._init_weights)

    self.attention_scores = {}

    # Attach hooks to save attention scores
    for idx, layer in enumerate(self.induction_heads):
      layer.register_forward_hook(self._save_attention_hook(idx))

  def _save_attention_hook(self, layer_id):
    def hook(module, input, output):
      # output contains (OV, attn_score) from Attn module's forward method
      _, attn_score = output
      self.attention_scores[layer_id] = attn_score.detach()  # avoid saving gradients
    return hook

  def _init_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
      module.weight.data.normal_(mean=0.0, std=0.02)
      if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

  def forward(self, x, target=None):
    embed_x = self.embed(x)

    if self.Position_mode =="learnable": 
      positions = self.pos_embed[:,:x.size(1),:]
      ihc_x = (embed_x+positions)
    elif self.Position_mode == "trigonometric":
      ihc_x = self.pos_embed(embed_x)
    else:
      ihc_x = embed_x
    

    for id, layer in enumerate(self.induction_heads):
      if self.norm_mode == "pre_norm":
        ihc_x = layer(self.norm(ihc_x))[0] + ihc_x
      elif self.norm_mode == "post_norm":
        ihc_x = self.norm(layer(ihc_x)[0]) + ihc_x
      else:
        ihc_x = layer(ihc_x)[0] + ihc_x

    unembed_x = self.unembed(ihc_x)
    
    loss = 0
    if target is not None:
      loss = F.cross_entropy(unembed_x.view(-1, unembed_x.size(-1)), target.view(-1))
    return unembed_x, loss