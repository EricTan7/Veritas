from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.utils.checkpoint import checkpoint
from .adapters import Conv_Adapter_v1, Conv_Adapter_v2, Conv_Adapter_v3, Conv_Adapter_v4, \
    Conv_Adapter_v5, Conv_Adapter_v6, Conv_Adapter_v7, Conv_Adapter_v8, Conv_Adapter_v9, \
    Conv_Adapter_v10, Conv_Adapter_v11, Conv_Adapter_v12, Conv_Adapter_v13, Conv_Adapter_v14
import ipdb


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    


class ModifiedAttention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x, num_loc_vpt=-1):
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # original self-attention for the original path
        # print("calculating attn_or")
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.out_proj(x))
    
        return x
    


class ModifiedAttention_vv(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x, num_loc_vpt=-1):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        # print("calculating attn_or")
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.out_proj(x))
        
        # self-attention, higher temperate for resnets performs better
        if num_loc_vpt > 0:
            v = v[:, :, :-num_loc_vpt]
        attn_new = (v @ v.transpose(-2, -1)) * self.scale
        attn_new = (attn_new).softmax(dim=-1)
        attn_new = self.attn_drop(attn_new)

        x_new = (attn_new @ v).transpose(1, 2).reshape(B, -1, C)
        x_new = self.proj_drop(self.out_proj(x_new))

        # return [x_new, x, attn_new, attn]
        return [x_new, x]


   

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, add_adapter=False, design_details=None):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor, text_fea=None):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out




class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, design_details=None):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        prompt_depth = design_details["vision_depth"]
        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0],
                                       add_adapter=True if 0 in prompt_depth else False,
                                       design_details=design_details)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2,
                                       add_adapter=True if 1 in prompt_depth else False,
                                       design_details=design_details)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2,
                                       add_adapter=True if 2 in prompt_depth else False,
                                       design_details=design_details)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2,
                                       add_adapter=True if 3 in prompt_depth else False,
                                       design_details=design_details)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    @torch.no_grad()
    def DPAM_replace(self):
        self.attnpool.DPAM_replace()
        # print("Not using DPAM!")

    @torch.no_grad()
    def Conv_Adapter_init(self):
        print("Not using Conv Adapter Init!")

    def _make_layer(self, planes, blocks, stride=1, add_adapter=False, design_details=None):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes, add_adapter=add_adapter, design_details=design_details))

        return nn.Sequential(*layers)

    def forward(self, x, features_layers=None, ffn=False, text_fea=None):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # [B, 2048, 14, 14]

        # Origin proj
        # x_cls = self.attnpool(x)     # [B, 512]
        # return x_cls.unsqueeze(1), x.flatten(2).transpose(1, 2)

        x = self.attnpool(x)
        # return x[0].unsqueeze(1), x[1:].transpose(0, 1)
        return x[:, :1], x[:, 1:]




class AttentionPool2d_org(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        # return x[0]
        return x



class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self.attn = None
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

    @torch.no_grad()
    def DPAM_replace(self):
        # reform transformer layer after init and load weights, using v only
        if self.attn == None:
            self.attn = ModifiedAttention(self.output_dim, self.embed_dim, self.num_heads, True)
            self.attn.qkv.weight = torch.nn.Parameter(torch.cat([self.v_proj.weight, self.v_proj.weight, self.v_proj.weight], 0))
            self.attn.qkv.bias = torch.nn.Parameter(torch.cat([self.v_proj.bias, self.v_proj.bias, self.v_proj.bias]))
            self.attn.out_proj.weight = self.c_proj.weight
            self.attn.out_proj.bias = self.c_proj.bias
            print("Using DPAM in the attnpool layer!")

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[0] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, x_ori = self.attn(x.transpose(0, 1))

        # cls token from the original path, and img tokens from the new path
        x[:, 0, :] = x_ori[:, 0, :]
        return x
    


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class ResidualAttentionBlock_IVLP(nn.Module):
    """
        Identical Vision Language Prompting (IVLP)
    """
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        # self.attn_mask = None

        # Only add learnable tokens if flag is set True
        # For the first iteration i, we should not add the learnable parameters
        # as it is already been taken care of in the very start, for both text
        # and the visual branch
        self.text_layer = text_layer
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_deep_embeddings = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, ModifiedAttention):
            return self.attn(x.transpose(0, 1)).transpose(0, 1)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, ffn=False, is_text=False, text_fea=None):
        # x: [n_tok, B, dim]
        # Will need to append the learnable tokens for this layer here
        # Check if flag was set for this layer or not
        if self.add_prompt:
            # Also see if this is textual transformer layer or not
            if not self.text_layer:
                # Remove the outputs produced by learnable tokens of previous layer
                prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
                # Create/configure learnable tokens of this layer
                visual_context = self.VPT_deep_embeddings.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, by replacing the previous
                # layer learnable tokens
                x = torch.cat([prefix, visual_context], dim=0)
            else:
                # Appending the learnable tokens in different way
                # x -> [77, NCLS, DIM]
                # First remove the learnable tokens from previous layer
                prefix = x[:1, :, :]
                suffix = x[1 + self.n_ctx_text:, :, :]
                # Create/configure learnable tokens of this layer
                textual_context = self.VPT_deep_embeddings.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                # Add the learnable tokens of this layer with the input, replaced by previous
                # layer learnable tokens
                x = torch.cat([prefix, textual_context, suffix], dim=0)

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x



class ResidualAttentionBlock_DPAM_IVLP(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, add_prompt=False,
                 text_layer=False, i=0, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.text_layer = text_layer
        if i != 0:
            self.add_prompt = add_prompt
            if self.add_prompt:
                if self.text_layer:
                    self.n_ctx_text = design_details["language_ctx"]
                    ctx_vectors = torch.empty(self.n_ctx_text, d_model)
                else:
                    self.n_ctx_visual = design_details["vision_ctx"]
                    ctx_vectors = torch.empty(self.n_ctx_visual, d_model)
                nn.init.normal_(ctx_vectors, std=0.02)
                self.VPT_deep_embeddings = nn.Parameter(ctx_vectors)
        else:
            self.add_prompt = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, ModifiedAttention_vv):
            x = x.transpose(0, 1)
            x_new, x = self.attn(x)
            return [x_new.transpose(0, 1), x.transpose(0, 1)]
        elif isinstance(self.attn, ModifiedAttention):
            return self.attn(x.transpose(0, 1)).transpose(0, 1)
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        
    def forward_add_prompt(self, x):
        if not self.text_layer:
            prefix = x[0:x.shape[0] - self.n_ctx_visual, :, :]
            # Create/configure learnable tokens of this layer
            visual_context = self.VPT_deep_embeddings.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
            # Add the learnable tokens of this layer with the input, by replacing the previous
            # layer learnable tokens
            x = torch.cat([prefix, visual_context], dim=0)
        else:
            # Appending the learnable tokens in different way
            # x -> [77, NCLS, DIM]
            # First remove the learnable tokens from previous layer
            prefix = x[:1, :, :]
            suffix = x[1 + self.n_ctx_text:, :, :]
            # Create/configure learnable tokens of this layer
            textual_context = self.VPT_deep_embeddings.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
            # Add the learnable tokens of this layer with the input, replaced by previous
            # layer learnable tokens
            x = torch.cat([prefix, textual_context, suffix], dim=0)
        return x

    def forward(self, x: torch.Tensor, ffn=False, is_text=False, text_fea=None):
        if isinstance(self.attn, ModifiedAttention_vv):
            if isinstance(x, list):
                x_new, x = x
            else:
                x_new, x = x, x
            if self.add_prompt:
                x = self.forward_add_prompt(x)
                x_new = self.forward_add_prompt(x_new)
            if not ffn:
                # x, x_ori = x
                # x_new_res, x_res, attn_new, attn = self.attention(self.ln_1(x))
                x_new_res, x_res = self.attention(self.ln_1(x))
                x = x + x_res
                x = x + self.mlp(self.ln_2(x))
                # skip FFN for new local feature 
                x_new = x_new + x_new_res    # x_new_res[:x_new.shape[0]]
                # return [x_new, x, attn_new, attn]
                return [x_new, x]
            else:
                x_new_res, x_res = self.attention(self.ln_1(x))
                x = x + x_res
                x = x + self.mlp(self.ln_2(x))
                # skip FFN for new local feature 
                x_new = x_new + x_new_res    # x_new_res[:x_new.shape[0]]
                x_new = x_new + self.mlp(self.ln_2(x_new))
                return [x_new, x]
        else:
            if self.add_prompt:
                x = self.forward_add_prompt(x)
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x





class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, prompts_needed=None,
                 text_layer=False, design_details=None, insert_text=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        current_trainer = design_details['trainer']

        if prompts_needed is None:
            # Origin: No prompting
            if current_trainer in ["DPAM", "IVLP"]:
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_DPAM_IVLP(width, heads, attn_mask, False, text_layer, i, design_details)
                    for i in range(layers)
                ])
            else:
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        else:
            if current_trainer == 'IVLP' or current_trainer == 'VPT':
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_IVLP(width, heads, attn_mask, True, text_layer, i, design_details) 
                    if i in prompts_needed else ResidualAttentionBlock_IVLP(width, heads, attn_mask, False, text_layer, i, design_details)
                    for i in range(layers)
                ])
            elif current_trainer == 'DPAM':
                self.resblocks = nn.Sequential(*[
                    ResidualAttentionBlock_DPAM_IVLP(width, heads, attn_mask, True, text_layer, i, design_details) 
                    if i in prompts_needed else ResidualAttentionBlock_DPAM_IVLP(width, heads, attn_mask, False, text_layer, i, design_details)
                    for i in range(layers)
                ])
            else:
                # Corresponds to default CoOp or CoCoOp
                assert current_trainer == 'CoOp' or current_trainer == 'CoCoOp'
                self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.embed_dim = width
        self.num_heads = heads
        self.attn = None

    @torch.no_grad()
    def attn_replace(self):
        for i in range(self.layers):
            self.attn = ModifiedAttention(self.embed_dim, self.embed_dim, self.num_heads, True)
            self.attn.q_proj.weight.data = self.resblocks[i].attn.in_proj_weight[:self.embed_dim].clone()
            self.attn.q_proj.bias.data = self.resblocks[i].attn.in_proj_bias[:self.embed_dim].clone()
            self.attn.k_proj.weight.data = self.resblocks[i].attn.in_proj_weight[self.embed_dim:self.embed_dim*2].clone()
            self.attn.k_proj.bias.data = self.resblocks[i].attn.in_proj_bias[self.embed_dim:self.embed_dim*2].clone()
            self.attn.v_proj.weight.data = self.resblocks[i].attn.in_proj_weight[self.embed_dim*2:].clone()
            self.attn.v_proj.bias.data = self.resblocks[i].attn.in_proj_bias[self.embed_dim*2:].clone()
            self.attn.out_proj.weight.data = self.resblocks[i].attn.out_proj.weight.clone()
            self.attn.out_proj.bias.data = self.resblocks[i].attn.out_proj.bias.clone()
            self.resblocks[i].attn = self.attn
        print(f"Textual attn replace Done: {list(range(self.layers))}")

    def CLIP_forward(self, x, out_layers=[11]):
        # Return: 最后一层origin features，中间层features
        # 不使用vv-attention, 但是返回中间层的feature
        out_tokens = []
        if out_layers is None:
            out_layers = [len(self.resblocks)-1]
        for idx, block in enumerate(self.resblocks):
            x = block(x)
            if idx in out_layers:
                out_tokens.append(x)
        return x, out_tokens
    
    def AnomalyCLIP_forward(self, x, out_layers=[11], ffn=False, text_fea=None):
        # Return: 最后一层origin features, 中间层vv-attn features
        # 使用vv-attention
        out_tokens = []
        if out_layers is None:
            out_layers = [len(self.resblocks)-1]
        for idx, block in enumerate(self.resblocks):
            if idx == 0:
                # x_new, x = block(x, ffn=ffn)
                x_new, x = block(x, ffn=ffn, text_fea=text_fea)
                # x_new, x, attn_new, attn = block(x, ffn = ffn)
            else:
                # x_new, x = block([x_new, x], ffn=ffn)
                x_new, x = block([x_new, x], ffn=ffn, text_fea=text_fea)
                # x_new, x, attn_new, attn = block([x_new, x], ffn = ffn)
            if idx in out_layers:
                out_tokens.append(x_new)
        return x, out_tokens
        # return x, out_tokens, attn_new, attn

    def forward(self, x, features_layers=[11], use_DPAM=False, ffn=False, use_checkpoint=False, text_fea=None):
        # text encoder forward
        if self.text_layer:
            if use_checkpoint:
                x = checkpoint(self.resblocks[:6], x)
                x = checkpoint(self.resblocks[6:], x)
                # x = checkpoint(self.resblocks[:3], x)
                # x = checkpoint(self.resblocks[3:6], x)
                # x = checkpoint(self.resblocks[6:9], x)
                # x = checkpoint(self.resblocks[9:], x)
                return x
            else:
                return self.resblocks(x)

        # visual encoder forward
        else:
            if use_DPAM:
                # print("AnomalyCLIP_foward")
                return self.AnomalyCLIP_forward(x, features_layers, ffn, text_fea)
            else:
                # print("CLIP_foward")
                return self.CLIP_forward(x, features_layers)
                



class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int,
                 output_dim: int, design_details):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        prompt_depth = design_details["vision_depth"]   # [-1] or [2,3,5,...]
        assert prompt_depth[-1] <= layers-1, "Prompt depth {} should not exceed Max layers {}".format(prompt_depth, layers)
        self.prompt_depth = prompt_depth if prompt_depth[0]!=-1 else None

        DPAM_layer = design_details["DPAM_layer"]   # [-1] or [2,3,5,...]
        assert DPAM_layer[-1] <= layers-1, "DPAM depth {} should not exceed Max layers {}".format(DPAM_layer, layers)
        self.DPAM_layer = DPAM_layer if DPAM_layer[0]!=-1 else None

        # self.VPT_shallow = True if self.prompt_depth is not None else False
        # self.VPT_shallow = True if self.prompt_depth is not None or self.prompt_depth[0] == 0 else False
        self.VPT_shallow = True if prompt_depth[0] == 0 else False

        if self.VPT_shallow:
            # Add visual prompt tokens here
            n_ctx = design_details["vision_ctx"]  # hyperparameter
            ctx_vectors = torch.empty(n_ctx, width)
            nn.init.normal_(ctx_vectors, std=0.02)
            self.VPT_embeddings = nn.Parameter(ctx_vectors)
            # self.VPT.half()
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        
        self.transformer = Transformer(width, layers, heads, prompts_needed=self.prompt_depth,
                                       design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.attn = None
        self.embed_dim = width
        self.num_heads = heads
        self.layers = layers

    @torch.no_grad()
    def DPAM_replace(self):
        if self.DPAM_layer is not None:
            for i in self.DPAM_layer:
                # out_proj.weight, out_proj.bias
                self.attn = ModifiedAttention_vv(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[i].attn.in_proj_bias.clone()
                self.attn.out_proj.weight.data = self.transformer.resblocks[i].attn.out_proj.weight.clone()
                self.attn.out_proj.bias.data = self.transformer.resblocks[i].attn.out_proj.bias.clone()
                self.transformer.resblocks[i].attn = self.attn
            print(f"DPAM replace Done: {self.DPAM_layer}")
        else:
            print("Not using DPAM!")

    @torch.no_grad()
    def Conv_Adapter_init(self, model=None):
        print(f"No Adapter!")

    @torch.no_grad()
    def attn_replace(self):
        for i in range(self.layers):
            self.attn = ModifiedAttention(self.embed_dim, self.embed_dim, self.num_heads, True)
            self.attn.q_proj.weight.data = self.transformer.resblocks[i].attn.in_proj_weight[:self.embed_dim].clone()
            self.attn.q_proj.bias.data = self.transformer.resblocks[i].attn.in_proj_bias[:self.embed_dim].clone()
            self.attn.k_proj.weight.data = self.transformer.resblocks[i].attn.in_proj_weight[self.embed_dim:self.embed_dim*2].clone()
            self.attn.k_proj.bias.data = self.transformer.resblocks[i].attn.in_proj_bias[self.embed_dim:self.embed_dim*2].clone()
            self.attn.v_proj.weight.data = self.transformer.resblocks[i].attn.in_proj_weight[self.embed_dim*2:].clone()
            self.attn.v_proj.bias.data = self.transformer.resblocks[i].attn.in_proj_bias[self.embed_dim*2:].clone()

            # self.attn.qkv.weight.data = self.transformer.resblocks[i].attn.in_proj_weight.clone()
            # self.attn.qkv.bias.data = self.transformer.resblocks[i].attn.in_proj_bias.clone()
            self.attn.out_proj.weight.data = self.transformer.resblocks[i].attn.out_proj.weight.clone()
            self.attn.out_proj.bias.data = self.transformer.resblocks[i].attn.out_proj.bias.clone()
            self.transformer.resblocks[i].attn = self.attn
        print(f"attn replace Done: {list(range(self.layers))}")

    def forward(self, x: torch.Tensor, features_layers=None, ffn=False, text_fea=None, return_loc=False, mid_proj=True):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)

        # After positional embeddings, we will attach prompts with the model, remember only those
        # are trainable parameters here in whole image encoder.
        if self.VPT_shallow:
            visual_ctx = self.VPT_embeddings.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)

        # Normal code as before
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        use_DPAM = True if self.DPAM_layer is not None else False
        # if self.DPAM_layer is not None:
            # features_layers = self.DPAM_layer
        # x, mid_features, attn_new, attn = self.transformer(x, features_layers, use_DPAM=use_DPAM, ffn=ffn) 
        x, mid_features = self.transformer(x, features_layers, use_DPAM=use_DPAM, ffn=ffn, text_fea=text_fea) 
        x = self.ln_post(x.permute(1, 0, 2))
        if return_loc:
            if mid_proj:
                mid_features_proj = []
                for fea in mid_features:
                    fea = self.ln_post(fea.permute(1, 0, 2)) @ self.proj  # LND -> NLD
                    mid_features_proj.append(fea)
            else:
                mid_features_proj = [fea.permute(1, 0, 2) for fea in mid_features]

            # return x[:, :1, :] @ self.proj, mid_features_proj, attn_new, attn
            return x[:, 0, :] @ self.proj, mid_features_proj
        else:
            return x[:, 0, :] @ self.proj




class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details
                 ):
        super().__init__()

        self.context_length = context_length
        trainer = design_details['trainer']

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                design_details=design_details
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                design_details=design_details
            )

        prompt_depth = design_details["language_depth"] # [-1] or [2,3,5,...]
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_depth,
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, features_layers=None, ffn=False):
        return self.visual(image.type(self.dtype), features_layers, ffn)

    def encode_text(self, text, use_checkpoint=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, use_checkpoint=use_checkpoint)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
    
    def encode_text_prompt(self, prompts, tokenized_prompts, deep_compound_prompts_text=None, normalize=False):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)[:, 0, :]
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        # logit_scale = torch.tensor(11.)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text



def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, input_size: list, design_details):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]

         # NOTE: resize pos_embed, so that input resolution can be customized
        if input_size[0] >= 224 and input_size[1] >= 224:
            num_x = input_size[1] // vision_patch_size
            num_y = input_size[0] // vision_patch_size
            num_patches = num_x * num_y
            if state_dict["visual.positional_embedding"].shape[0] != num_patches + 1:
                state_dict["visual.positional_embedding"] = resize_pos_embed(state_dict["visual.positional_embedding"], 
                                                            num_patches+1, num_y, num_x)
                
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]


        # assert input_size[0] == input_size[1]
        if input_size[0] >= 224 and input_size[1] >= 224:
            num_x = input_size[0] // 32
            num_fea = num_x ** 2
            if state_dict["visual.attnpool.positional_embedding"].shape[0] != num_fea + 1:
                state_dict["visual.attnpool.positional_embedding"] = resize_pos_embed(state_dict["visual.attnpool.positional_embedding"], 
                                                            num_fea+1, num_x, num_x)
                
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model.eval()


def resize_pos_embed(posemb, num_tok, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    # posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    num_tok -= 1
    posemb_token, posemb_grid = posemb[0, :], posemb[1:, :]

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from:{} to: {} with height:{} width: {}'.format(posemb.shape[0], num_tok, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb_grid = posemb_grid.squeeze(0)
    posemb_token = posemb_token.unsqueeze(0)
    posemb = torch.cat([posemb_token, posemb_grid], dim=0)
    return posemb
