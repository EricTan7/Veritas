import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp
from networks.clip import clip
from networks.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from typing import Callable, List, Optional, Sequence, Tuple, Union
# from visualizer import get_local
# get_local.activate()

_tokenizer = _Tokenizer()
device = 'cuda'



def get_template_base_fea(clip_model, classnames):
    base_template = "A photo of a {}."
    base_text = [base_template.format(name) for name in classnames]
    base_text_tokens = clip.tokenize(base_text)
    with torch.no_grad():
        base_text_fea = clip_model.encode_text(base_text_tokens)  
    return base_text_fea.unsqueeze(1).detach().to("cuda")     # [attr, 1, dim]


def update_prompt_misc(cfg, llm, clip_model, classnames):
    if cfg.SOLVER.DEBUG:
        # llm.llm_model.to("cpu")
        text_templates = "A photo of a {}."
        text_templates = [text_templates.format(classnames[i]) for i in range(len(classnames))]
    else:
        class_descrips = llm()
        text_templates = "A photo of a {}."
        text_templates = [class_descrips[i] + ' ' + text_templates.format(classnames[i]) for i in range(len(classnames))]
    # text_templates = [text_templates.format(classnames[i]) + ' ' + class_descrips[i]  for i in range(len(classnames))]
    print(text_templates[1])
    print(text_templates[11])
    text_tok = clip.tokenize(text_templates)
    with torch.no_grad():
        text_fea = clip_model.encode_text(text_tok)
    return text_fea.unsqueeze(1).detach().to("cuda")


   

def exists(val):
    return val is not None



class InversionNetwork(nn.Module):
    def __init__(self, in_dim, out_dim=None, reduction=16):
        super().__init__()
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_dim, in_dim // reduction)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(in_dim // reduction, out_dim))
        ]))

    def forward(self, x):
        return self.net(x)
    

class InversionNetwork_LN(nn.Module):
    def __init__(self, in_dim, out_dim=None, reduction=16):
        super().__init__()
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_dim, in_dim // reduction)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(in_dim // reduction, out_dim)),
            ("norm", nn.LayerNorm(out_dim))
        ]))

    def forward(self, x):
        return self.net(x)



class FeedForward(nn.Module):
    """
        MLP with pre-norm
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

        self.norm = nn.LayerNorm(in_features)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x



class ResamplerAttention(nn.Module):
    """
        Multi-head cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm_x = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x, kv, padding_mask=None):
        # x: learnable query
        # kv: text embeddings
        B, N1, C = x.shape
        _, N2, _ = kv.shape
        x = self.norm_x(x)
        kv = self.norm_kv(kv)

        q = self.to_q(x).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_input = self.to_kv(kv).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv_input[0], kv_input[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if padding_mask is not None:
            if len(padding_mask.shape) == 2:
                padding_mask = padding_mask.unsqueeze(1)
                padding_mask = padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            fill_value = -1e9 if attn.dtype==torch.float32 else -1e4
            attn = attn.masked_fill(padding_mask == 0, fill_value)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class FocusResampler(nn.Module):
    def __init__(
        self,
        dim,
        depth=6,
        heads=8,
        mlp_ratio=4.,
        num_query=64,
        max_num_tok=None
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_query, dim))
        self.time_embs = (
            nn.Parameter(torch.randn(max_num_tok, 1, dim))
            if exists(max_num_tok)
            else None
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ResamplerAttention(dim=dim, num_heads=heads),
                        FeedForward(in_features=dim, hidden_features=mlp_hidden_dim),
                    ]
                )
            )

    def forward(self, x, padding_mask=None):
        """
        Args:
            x (torch.Tensor): text embeddings [N_CLS, N_TOK, DIM]
        Returns:
            [N_CLS, 1, DIM]
        """
        B, N, C = x.shape

        if exists(self.time_embs):
            x = x + self.time_embs[:N]

        # blocks
        query = self.query.unsqueeze(0).expand(B, -1, -1)
        for attn, ff in self.layers:
            query = attn(query, x, padding_mask) + query
            query = ff(query) + query
        return query


class FeedForwardSwiGLU(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForwardSwiGLU, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc2 = nn.Linear(embed_size, ff_hidden_dim)
        self.fc3 = nn.Linear(ff_hidden_dim, embed_size)

    def forward(self, x):
        gate = torch.sigmoid(self.fc1(x))
        transformed = torch.relu(self.fc2(x))
        output = gate * transformed
        return self.fc3(output)
    

def shuffle_unit(features, shift, group, begin=1):
    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Block(nn.Module):
    """
    Transformer Encoder Block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
  

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # nn.init.eye_(self.q.weight.data)

    # @get_local('attn')
    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        query = self.q(query)
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        # query = self.proj(query)
        query = self.proj_drop(query)

        return query

    
class CrossAttention_norm(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

    # @get_local('attn')
    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]
        query = self.norm_q(query)
        kv = self.norm_kv(kv)

        # 1. project q, insert a learnable layer
        query = self.q(query)
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        query = self.proj(query)
        query = self.proj_drop(query)

        return query
    


class MultiHeadCrossAttention(nn.Module):
    """
        Multi-head cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        # self.to_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv):
        B, N1, C = x.shape
        _, N2, _ = kv.shape

        q = self.to_q(x).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # kv = self.to_kv(kv).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]
        kv = kv.reshape(B, N2, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k, v = kv, kv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class Gated_CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear1 = nn.Linear(dim*4, dim, bias=True)
        self.linear2 = nn.Linear(dim*4, dim, bias=True)

        # self.linear1 = nn.Linear(dim, dim, bias=True)
        # self.linear2 = nn.Linear(dim, dim, bias=True)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)

    # @get_local('attn')
    # @get_local('gate_value')
    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        query = self.q(query)
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query_o = attn @ v    # [B, n_query, dim]
        query_o = self.proj(query_o)
        query_o = self.proj_drop(query_o)

        modulated_o = F.tanh(self.linear1(torch.cat([query_o, query, query_o-query, query_o*query], dim=-1)))
        gate_value = F.sigmoid(self.linear2(torch.cat([query_o, query, query_o-query, query_o*query], dim=-1)))

        # modulated_o = F.tanh(self.linear1(query_o))
        # gate_value = F.sigmoid(self.linear2(query_o-query))

        gate_o = gate_value * modulated_o + (1 - gate_value) * query

        return gate_o
    


class CrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttnBlock, self).__init__()
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.norm_mlp = norm_layer(dim)

    def forward(self, query, kv):
        # query: [B, attr, 512]
        # kv: [B, len, 512]

        # LN + cross-attn
        query = query + self.cross_attn(self.norm_q(query), self.norm_kv(kv))

        # LN + MLP
        query = query + self.mlp(self.norm_mlp(query))

        return query
    


class CrossAttn(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CrossAttn, self).__init__()
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)

    def forward(self, query, kv):
        # query: [B, attr, 512]
        # kv: [B, len, 512]

        # LN + cross-attn
        query = query + self.cross_attn(self.norm_q(query), self.norm_kv(kv))

        return query



class MultiHeadCrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(MultiHeadCrossAttnBlock, self).__init__()
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.norm_mlp = norm_layer(dim)

    def forward(self, query, kv):
        # query: [B, attr, 512]
        # kv: [B, len, 512]

        # LN + cross-attn
        query = query + self.cross_attn(self.norm_q(query), self.norm_kv(kv))

        # LN + MLP
        query = query + self.mlp(self.norm_mlp(query))

        return query
    


class GatedCrossAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(GatedCrossAttnBlock, self).__init__()
        self.cross_attn = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.norm_mlp = norm_layer(dim)

    def forward(self, query, kv):
        query = query + self.cross_attn(self.norm_q(query), self.norm_kv(kv))
        query = query + self.mlp(self.norm_mlp(query))

        return query



class CoAttnBlock(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(CoAttnBlock, self).__init__()
        self.cross_attn1 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        # self.cross_attn1 = CrossAttention_projkv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)
        # self.cross_attn2 = CrossAttention_projkv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)

        self.mlp_text = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp_img = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm_cross_text = norm_layer(dim)
        self.norm_cross_img = norm_layer(dim)
        self.norm_mlp_text = norm_layer(dim)
        self.norm_mlp_img = norm_layer(dim)

    def forward(self, text_fea, img_fea):
        # text_fea: [B, attr, 512]
        # img_fea: [B, len, 512]

        # LN + cross-attn
        text_fea = text_fea + self.cross_attn1(self.norm_cross_text(text_fea), self.norm_cross_img(img_fea))
        img_fea = img_fea + self.cross_attn2(self.norm_cross_img(img_fea), self.norm_cross_text(text_fea))

        # LN + MLP
        text_fea = text_fea + self.mlp_text(self.norm_mlp_text(text_fea))
        img_fea = img_fea + self.mlp_img(self.norm_mlp_img(img_fea))

        return text_fea, img_fea
    


class AsyCoAttnBlock(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(AsyCoAttnBlock, self).__init__()
        self.cross_attn1 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        # self.cross_attn1 = CrossAttention_projkv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)
        # self.cross_attn2 = CrossAttention_projkv(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)

        self.mlp_text = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp_img = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm_cross_text = norm_layer(dim)
        self.norm_cross_img = norm_layer(dim)
        self.norm_mlp_text = norm_layer(dim)
        self.norm_mlp_img = norm_layer(dim)

    def forward(self, text_fea, img_fea, img_fea2):
        # text_fea: [B, attr, 512]
        # img_fea: [B, len, 512]

        # LN + cross-attn
        text_fea = text_fea + self.cross_attn1(self.norm_cross_text(text_fea), self.norm_cross_img(img_fea))
        img_fea2 = img_fea2 + self.cross_attn2(self.norm_cross_img(img_fea2), self.norm_cross_text(text_fea))

        # LN + MLP
        text_fea = text_fea + self.mlp_text(self.norm_mlp_text(text_fea))
        img_fea2 = img_fea2 + self.mlp_img(self.norm_mlp_img(img_fea2))

        return text_fea, img_fea2
    


class MultiHeadCoAttnBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(MultiHeadCoAttnBlock, self).__init__()
        self.cross_attn1 = MultiHeadCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = MultiHeadCrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.mlp_text = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        self.mlp_img = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)

        self.norm_cross_text = norm_layer(dim)
        self.norm_cross_img = norm_layer(dim)
        self.norm_mlp_text = norm_layer(dim)
        self.norm_mlp_img = norm_layer(dim)

    def forward(self, text_fea, img_fea):
        # text_fea: [B, attr, 512]
        # img_fea: [B, len, 512]

        # LN + cross-attn
        text_fea = text_fea + self.cross_attn1(self.norm_cross_text(text_fea), self.norm_cross_img(img_fea))
        img_fea = img_fea + self.cross_attn2(self.norm_cross_img(img_fea), self.norm_cross_text(text_fea))

        # LN + MLP
        text_fea = text_fea + self.mlp_text(self.norm_mlp_text(text_fea))
        img_fea = img_fea + self.mlp_img(self.norm_mlp_img(img_fea))

        return text_fea, img_fea



class Gated_CoAttnBlock(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(Gated_CoAttnBlock, self).__init__()
        self.cross_attn1 = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.mlp_text = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp_img = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm_cross_text = norm_layer(dim)
        self.norm_cross_img = norm_layer(dim)
        self.norm_mlp_text = norm_layer(dim)
        self.norm_mlp_img = norm_layer(dim)

    def forward(self, text_fea, img_fea):
        # text_fea: [B, attr, 512]
        # img_fea: [B, len, 512]

        # LN + cross-attn
        text_fea = text_fea + self.cross_attn1(self.norm_cross_text(text_fea), self.norm_cross_img(img_fea))
        img_fea = img_fea + self.cross_attn2(self.norm_cross_img(img_fea), self.norm_cross_text(text_fea))

        # LN + MLP
        text_fea = text_fea + self.mlp_text(self.norm_mlp_text(text_fea))
        img_fea = img_fea + self.mlp_img(self.norm_mlp_img(img_fea))

        return text_fea, img_fea



class AsyGated_CoAttnBlock(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(AsyGated_CoAttnBlock, self).__init__()
        self.cross_attn1 = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.mlp_text = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp_img = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm_cross_text = norm_layer(dim)
        self.norm_cross_img = norm_layer(dim)
        self.norm_mlp_text = norm_layer(dim)
        self.norm_mlp_img = norm_layer(dim)

    def forward(self, text_fea, img_fea, img_fea2):
        # text_fea: [B, attr, 512]
        # img_fea: [B, len, 512]

        # LN + cross-attn
        text_fea = text_fea + self.cross_attn1(self.norm_cross_text(text_fea), self.norm_cross_img(img_fea))
        img_fea2 = img_fea2 + self.cross_attn2(self.norm_cross_img(img_fea2), self.norm_cross_text(text_fea))

        # LN + MLP
        text_fea = text_fea + self.mlp_text(self.norm_mlp_text(text_fea))
        img_fea2 = img_fea2 + self.mlp_img(self.norm_mlp_img(img_fea2))

        return text_fea, img_fea2



class Gated_CoAttnBlock_parallel(nn.Module):
    """
    input: embedded_tokens
            (x)  tokens -> projection(Linear), project to new Q,K,V -> LN + Cross-Attention + LN + MLP
            (√)  tokens -> LN + self-attn -> LN + cross-attn -> LN + MLP
          residual:     |------------------| |--------------| |--------|
    output: tokens
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super(Gated_CoAttnBlock_parallel, self).__init__()
        self.cross_attn1 = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.cross_attn2 = Gated_CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                          attn_drop=attn_drop, proj_drop=drop)
        self.mlp_text = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)
        self.mlp_img = Mlp(in_features=dim, hidden_features=dim, act_layer=act_layer, drop=drop)

        self.norm_cross_text = norm_layer(dim)
        self.norm_cross_img = norm_layer(dim)
        self.norm_mlp_text = norm_layer(dim)
        self.norm_mlp_img = norm_layer(dim)

    def forward(self, text_fea, img_fea):
        # text_fea: [B, attr, 512]
        # img_fea: [B, len, 512]

        # LN + cross-attn
        text_fea_org = text_fea.clone()
        img_fea_text = img_fea.clone()

        text_fea = text_fea + self.cross_attn1(self.norm_cross_text(text_fea), self.norm_cross_img(img_fea_text))
        img_fea = img_fea + self.cross_attn2(self.norm_cross_img(img_fea), self.norm_cross_text(text_fea_org))

        # LN + MLP
        text_fea = text_fea + self.mlp_text(self.norm_mlp_text(text_fea))
        img_fea = img_fea + self.mlp_img(self.norm_mlp_img(img_fea))

        return text_fea, img_fea



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # todo: the same as CLIP
        # positional_embedding
        x = prompts + self.positional_embedding.type(self.dtype)
        # x = prompts
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x
    


class TextEncoder_remoteclip(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.attn_mask = clip_model.attn_mask.to(device)
        self.text_pool_type = clip_model.text_pool_type

    def forward(self, prompts, tokenized_prompts, use_checkpoint=False):
        # todo: the same as CLIP
        # positional_embedding
        x = prompts + self.positional_embedding.type(self.dtype)
        # x = prompts
        # x = x.permute(1, 0, 2)  # NLD -> LND
        # print(x.device)
        # print(self.attn_mask.device)
        # print(self.text_projection.device)
        # print(self.transformer.device)
        x = self.transformer(x, attn_mask=self.attn_mask)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x, _ = text_global_pool(x, tokenized_prompts, self.text_pool_type)
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection

        return x



def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextEncoder_PE(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # positional_embedding = torch.cat([self.positional_embedding[1:], self.positional_embedding[:1]], dim=0)
        x = x + self.positional_embedding.type(self.dtype)
        # x = torch.cat([x[:, 1:], x[:, :1]], dim=1)
        # x = x + positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, attn = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, attn
   


class Prompter(nn.Module):
    # init prompt
    # get prompted features
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.INPUT.PROMPT.N_CTX
        ctx_init = cfg.INPUT.PROMPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.INPUT.PROMPT.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        # NOTE: buffer - no grad      SOS, EOS, CLASS
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # NOTE: the following is not "Parameters", (1) will not be saved when in save_model(), (2) do not have grads
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.INPUT.PROMPT.CLASS_TOKEN_POSITION

        self.text_encoder = TextEncoder(clip_model)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        prompt_features = self.text_encoder(prompts, self.tokenized_prompts)

        return prompt_features


def normalize(logit, dim=-1):
    mean = logit.mean(dim=dim, keepdims=True)
    stdv = logit.std(dim=dim, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


class SoftmaxWithTemperature(nn.Module):
    def __init__(self, initial_temperature=1.0, ln=True):
        super(SoftmaxWithTemperature, self).__init__()
        if ln:
            self.logit_scale_temperature = nn.Parameter(torch.tensor(initial_temperature))
        else:
            self.logit_scale_temperature = torch.tensor(initial_temperature)

    def forward(self, logits, dim=-1):
        scaled_logits = logits * self.logit_scale_temperature
        softmax_output = torch.softmax(scaled_logits, dim=dim)
        return softmax_output
    

class LNLogitScale(nn.Module):
    def __init__(self, init, ln=False):
        super(LNLogitScale, self).__init__()
        if ln:
            self.ls = nn.Parameter(init)
        else:
            self.ls = init

    def forward(self, img, text):
        return self.ls * img @ text
    

class LNLogitScaleCSC(nn.Module):
    def __init__(self, init, n_cls, ln=False):
        super(LNLogitScaleCSC, self).__init__()
        if ln:
            self.ls = nn.Parameter(torch.full((1, n_cls), init))
        else:
            self.ls = torch.full((1, n_cls), init)
        
        self.n_cls = n_cls

    def forward(self, img, text):
        logits = img @ text
        logits = logits.view(img.size(0), 1, self.n_cls, -1)   # [B, 1, attr, groups]
        ls = self.ls.unsqueeze(0).unsqueeze(-1).expand(img.size(0), -1, -1, logits.size(-1))  # [B, 1, attr, groups]
        return ls * logits


class LNLogitScaleCSC2(nn.Module):
    def __init__(self, init, n_cls, ln=False):
        super(LNLogitScaleCSC2, self).__init__()
        if ln:
            self.ls = nn.Parameter(torch.full((1, 1, n_cls), init))
        else:
            self.ls = torch.full((1, 1, n_cls), init)

        self.n_cls = n_cls

    def forward(self, img, text):   # text [B, dim, attr*group]
        ls = self.ls.repeat(img.size(0), 1, text.size(2)//self.n_cls)
        return ls * (img @ text)




class FocusResampler_nocat(nn.Module):
    def __init__(
        self,
        dim,
        depth=6,
        heads=8,
        mlp_ratio = 4.,
        num_query=64,
        max_num_tok=None
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_query, dim))
        self.time_embs = (
            nn.Parameter(torch.randn(max_num_tok, 1, dim))
            if exists(max_num_tok)
            else None
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.layers = nn.ModuleList()
        # for _ in range(depth):
        #     self.layers.append(
        #         nn.Sequential(OrderedDict([
        #             ("attn", ResamplerAttention(dim=dim)),
        #             ("mlp", FeedForward(in_features=dim, hidden_features=mlp_hidden_dim))
        #         ]))
        #     )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ResamplerAttention_nocat(dim=dim, num_heads=heads),
                        FeedForward(in_features=dim, hidden_features=mlp_hidden_dim),
                    ]
                )
            )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): text embeddings [N_CLS, N_TOK, DIM]
        Returns:
            [N_CLS, 1, DIM]
        """
        B, N, C = x.shape

        if exists(self.time_embs):
            x = x + self.time_embs[:N]

        # blocks
        query = self.query.unsqueeze(0).expand(B, -1, -1)
        for attn, ff in self.layers:
            query = attn(query, x) + query
            query = ff(query) + query
        return query




class FocusResampler_mask(nn.Module):
    def __init__(
        self,
        dim,
        depth=6,
        heads=8,
        mlp_ratio = 4.,
        num_query=64,
        max_num_tok=None
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(num_query, dim))
        self.time_embs = (
            nn.Parameter(torch.randn(max_num_tok, 1, dim))
            if exists(max_num_tok)
            else None
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.layers = nn.ModuleList()
        # for _ in range(depth):
        #     self.layers.append(
        #         nn.Sequential(OrderedDict([
        #             ("attn", ResamplerAttention(dim=dim)),
        #             ("mlp", FeedForward(in_features=dim, hidden_features=mlp_hidden_dim))
        #         ]))
        #     )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        ResamplerAttention_mask(dim=dim, num_heads=heads),
                        FeedForward(in_features=dim, hidden_features=mlp_hidden_dim),
                    ]
                )
            )

    def forward(self, x, padding_mask=None):
        """
        Args:
            x (torch.Tensor): text embeddings [N_CLS, N_TOK, DIM]
        Returns:
            [N_CLS, 1, DIM]
        """
        B, N, C = x.shape

        if exists(self.time_embs):
            x = x + self.time_embs[:N]

        # blocks
        query = self.query.unsqueeze(0).expand(B, -1, -1)
        for attn, ff in self.layers:
            query = attn(query, x, padding_mask) + query
            query = ff(query) + query
        return query





class ResamplerAttention_nocat(nn.Module):
    """
        Multi-head cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm_x = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x, kv):
        # x: learnable query
        # kv: text embeddings
        B, N1, C = x.shape
        _, N2, _ = kv.shape
        x = self.norm_x(x)
        kv = self.norm_kv(kv)

        # B, heads, N, C//num_heads
        q = self.to_q(x).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # kv_input = torch.cat([kv, x], dim=-2)
        kv_input = self.to_kv(kv).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv_input[0], kv_input[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
   



class ResamplerAttention_mask(nn.Module):
    """
        Multi-head cross-attention
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.norm_x = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, x, kv, padding_mask=None):
        # x: learnable query
        # kv: text embeddings
        B, N1, C = x.shape
        _, N2, _ = kv.shape
        x = self.norm_x(x)
        kv = self.norm_kv(kv)

        # B, heads, N, C//num_heads
        q = self.to_q(x).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # kv_input = torch.cat([kv, x], dim=-2)
        kv_input = self.to_kv(kv).reshape(B, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv_input[0], kv_input[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # if test:
        #     print("testing")
        #     if padding_mask is not None:
        #         print("padding_mask: ", padding_mask.shape)
        #     else:
        #         print("No padding mask!")
        #     print("\n\n")
        # else:
        #     print("training")
        #     if padding_mask is not None:
        #         print("padding_mask: ", padding_mask.shape)
        #     else:
        #         print("No padding mask!")
        #     print("\n\n")
        if padding_mask is not None:
            if len(padding_mask.shape) == 2:
                padding_mask = padding_mask.unsqueeze(1)
                padding_mask = padding_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            fill_value = -1e9 if attn.dtype==torch.float32 else -1e4
            attn = attn.masked_fill(padding_mask == 0, fill_value)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class CrossAttention_woproj(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.q = nn.Linear(dim, dim, bias=qkv_bias)
        # nn.init.eye_(self.q.weight.data)

    # @get_local('attn')
    def forward(self, query, kv):
        # query:[B, n_query, dim]
        # img_fea: [B, n_tkn, dim]

        # 1. project q, insert a learnable layer
        # query = self.q(query)
        k, v = kv, kv

        attn = (query @ k.transpose(1,2)) * self.scale   # [B, n_query, n_tkn]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        query = attn @ v    # [B, n_query, dim]
        # query = self.proj(query)
        query = self.proj_drop(query)

        return query
    


class TextSpatialFusion_v1(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        # self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)
        # # self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        # self.norm_q = norm_layer(dim)
        # self.norm_kv = norm_layer(dim)
        # self.norm_mlp = norm_layer(dim)

        # spatial fusion
        hidden_dim = dim // 2
        layers1 = [
            nn.Conv2d(dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=1, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=1, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        kernel_size=3
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=1//2, bias=False)

        self.adapter_norm_q = nn.LayerNorm(dim)
        self.adapter_norm_kv = nn.LayerNorm(dim)
        self.adapter_cattn = CrossAttention_woproj(dim)

    def forward(self, x, text_fea):
        # query: [B, attr, 512]
        # kv: [B, len, 512]

        # LN + cross-attn
        x = x + self.adapter_cattn(self.adapter_norm_q(x), self.adapter_norm_kv(text_fea))

        B, tok, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(0, 2, 1).reshape(B, dim, H, H)

        x_loc1 = self.conv_adapter_layers1(x_loc)
        x_loc2 = self.conv_adapter_layers2(x_loc)
        x_loc = torch.cat([x_loc1, x_loc2], dim=1)
        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
        agg = torch.cat([avg_x, max_x], dim=1)
        y = self.conv_adapter_layers(agg)
        # y = y.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]
        y = F.sigmoid(y)
        x = x_loc1 * y[:, 0].unsqueeze(1) + x_loc2 * y[:, 1].unsqueeze(1)
        x = self.conv_adapter_final(x)
        x = x.reshape(B, -1, tok).permute(0, 2, 1)  # [tok, B, 2]

        # LN + MLP
        # query = query + self.mlp(self.norm_mlp(query))

        return x
    





class TextSpatialFusion_v2(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        # self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)
        # self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        # self.norm_q = norm_layer(dim)
        # self.norm_kv = norm_layer(dim)
        # self.norm_mlp = norm_layer(dim)

        # spatial fusion
        hidden_dim = dim
        layers1 = [
            nn.Conv2d(dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=1, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=1, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        kernel_size=3
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(dim)
        self.adapter_norm_q2 = nn.LayerNorm(dim)
        self.adapter_norm_kv = nn.LayerNorm(dim)
        self.adapter_cattn = CrossAttention_woproj(dim)

    def forward(self, x, text_fea):
        # query: [B, attr, 512]
        # kv: [B, len, 512]

        B, tok, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(0, 2, 1).reshape(B, dim, H, H)

        x_loc1 = self.conv_adapter_layers1(x_loc)
        x_loc2 = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1.reshape(B, -1, tok)
        x_loc2 = x_loc2.reshape(B, -1, tok)

        x_loc1 = x_loc1 + self.adapter_cattn(
            self.adapter_norm_q1(x_loc1.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)
        x_loc2 = x_loc2 + self.adapter_cattn(
            self.adapter_norm_q2(x_loc2.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)

        x_loc1 = x_loc1.reshape(B, -1, H, H)
        x_loc2 = x_loc2.reshape(B, -1, H, H)

        x_loc = torch.cat([x_loc1, x_loc2], dim=1)
        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
        agg = torch.cat([avg_x, max_x], dim=1)
        y = self.conv_adapter_layers(agg)
        # y = y.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]
        y = F.sigmoid(y)
        x = x_loc1 * y[:, 0].unsqueeze(1) + x_loc2 * y[:, 1].unsqueeze(1)
        x = self.conv_adapter_final(x)
        x = x.reshape(B, -1, tok).permute(0, 2, 1)  # [tok, B, 2]

        # LN + MLP
        # query = query + self.mlp(self.norm_mlp(query))

        return x
    




class TextSpatialFusion_v3(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        # self.cross_attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                                   attn_drop=attn_drop, proj_drop=drop)
        # # self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)
        # self.norm_q = norm_layer(dim)
        # self.norm_kv = norm_layer(dim)
        # self.norm_mlp = norm_layer(dim)

        # spatial fusion
        hidden_dim = dim
        layers1 = [
            nn.Conv2d(dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=1, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=1, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        kernel_size=3
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, dim, kernel_size=1, stride=1, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(dim)
        self.adapter_norm_q2 = nn.LayerNorm(dim)
        self.adapter_norm_kv = nn.LayerNorm(dim)
        self.adapter_cattn = CrossAttention_woproj(dim)

    def forward(self, x, text_fea):
        # query: [B, attr, 512]
        # kv: [B, len, 512]

        B, tok, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(0, 2, 1).reshape(B, dim, H, H)

        x_loc1_bef = self.conv_adapter_layers1(x_loc)
        x_loc2_bef = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1_bef.reshape(B, -1, tok)
        x_loc2 = x_loc2_bef.reshape(B, -1, tok)

        x_loc1 = x_loc1 + self.adapter_cattn(
            self.adapter_norm_q1(x_loc1.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)
        x_loc2 = x_loc2 + self.adapter_cattn(
            self.adapter_norm_q2(x_loc2.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)

        x_loc1 = x_loc1.reshape(B, -1, H, H)
        x_loc2 = x_loc2.reshape(B, -1, H, H)

        x_loc = torch.cat([x_loc1, x_loc2], dim=1)
        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
        agg = torch.cat([avg_x, max_x], dim=1)
        y = self.conv_adapter_layers(agg)
        # y = y.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]
        y = F.sigmoid(y)
        x = x_loc1_bef * y[:, 0].unsqueeze(1) + x_loc2_bef * y[:, 1].unsqueeze(1)
        x = self.conv_adapter_final(x)
        x = x.reshape(B, -1, tok).permute(0, 2, 1)  # [tok, B, 2]

        # LN + MLP
        # query = query + self.mlp(self.norm_mlp(query))

        return x
    




