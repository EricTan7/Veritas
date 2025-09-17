from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from torch.utils.checkpoint import checkpoint
# from visualizer import get_local



class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# self.meta_net = nn.Sequential(OrderedDict([
#             ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
#             ("relu", nn.ReLU(inplace=True)),
#             ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
#         ]))
class Conv_Adapter_v1(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        1x1 Conv(point-wise conv) + 3x3 DW-Conv + 1x1 Conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        # hidden_dim = int(in_dim * expand_ratio)
        hidden_dim = in_dim if hidden_dim is None else hidden_dim
        kernel_size = kernel_size

        layers = []
        # 1x1 conv
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            QuickGELU()
            ])

        # 3x3 DW-conv
        dp = [
            nn.Conv2d(hidden_dim, 
                      hidden_dim, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=kernel_size//2, 
                      groups=hidden_dim, 
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True)
            QuickGELU()
        ]
        layers.extend(dp)

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True)
            QuickGELU()
        ])
        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x):
        x_cls, x = x[:1], x[1:]
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x = self.conv_adapter_layers(x)
        x = x.reshape(B, dim, tok).permute(2, 0, 1)
        x = torch.cat([x_cls, x], dim=0)
        return x
    




class Conv_Adapter_v2(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        1x1 Conv(point-wise conv) + 3x3 Conv + 1x1 Conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        # hidden_dim = int(in_dim * expand_ratio)
        hidden_dim = in_dim if hidden_dim is None else hidden_dim
        kernel_size = kernel_size

        layers = []
        # 1x1 conv
        layers.extend([
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            QuickGELU()
            ])

        # 3x3 conv
        dp = [
            nn.Conv2d(hidden_dim, 
                      hidden_dim, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=kernel_size//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True)
            QuickGELU()
        ]
        layers.extend(dp)

        # the second linear layer is replaced by 1x1 convolution.
        layers.extend([
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            QuickGELU()
        ])
        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x):
        x_cls, x = x[:1], x[1:]
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x = self.conv_adapter_layers(x)
        x = x.reshape(B, dim, tok).permute(2, 0, 1)
        x = torch.cat([x_cls, x], dim=0)
        return x
    


class Conv_Adapter_v3(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        3x3 DW-Conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        kernel_size = kernel_size
        layers = []

        # 3x3 conv
        dp = [
            nn.Conv2d(in_dim, 
                      out_dim, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=kernel_size//2,  
                      groups=in_dim, 
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True)
            QuickGELU()
        ]
        layers.extend(dp)

        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x):
        x_cls, x = x[:1], x[1:]
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x = self.conv_adapter_layers(x)
        x = x.reshape(B, dim, tok).permute(2, 0, 1)
        x = torch.cat([x_cls, x], dim=0)
        return x
    


class Conv_Adapter_v4(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        3x3 Conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        kernel_size = kernel_size
        layers = []

        # 3x3 conv
        dp = [
            nn.Conv2d(in_dim, 
                      out_dim, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=kernel_size//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            QuickGELU()
        ]
        layers.extend(dp)

        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x, proj=False):
        if proj:
            return self.forward_proj(x)
        x_cls, x = x[:1], x[1:]
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x = self.conv_adapter_layers(x)
        x = x.reshape(B, -1, tok).permute(2, 0, 1)
        x = torch.cat([x_cls, x], dim=0)
        return x

    def forward_proj(self, x):
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x = self.conv_adapter_layers(x)
        x = x.reshape(B, -1, tok).permute(2, 0, 1)
        return x
    





class Conv_Adapter_v5(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        ECA_combined, max and min use the same conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        layers = [
            nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        ]

        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x, proj=False):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
        avg_x = self.conv_adapter_layers(avg_x)
        max_x = self.conv_adapter_layers(max_x)
        y = avg_x + max_x
        y = y.reshape(B, -1, tok).permute(2, 0, 1)
        y = F.sigmoid(y)
        x = x * y.expand_as(x)
        
        x = torch.cat([x_cls, x], dim=0)
        return x





class Conv_Adapter_v6(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        max and min first concat, then conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        layers = [
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        ]

        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x, proj=False):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
        y = torch.cat([avg_x, max_x], dim=1)
        y = self.conv_adapter_layers(y)
        y = y.reshape(B, -1, tok).permute(2, 0, 1)
        y = F.sigmoid(y)
        x = x * y.expand_as(x)
        
        x = torch.cat([x_cls, x], dim=0)
        return x




class Conv_Adapter_v7(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3, **kwargs):
        """
        LSKNet
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        hidden_dim = in_dim//2 if hidden_dim is None else hidden_dim
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)
        
    def forward(self, x, proj=False, text_fea=None, channel_first=True):
        if channel_first:
            if proj:
                return self.forward_proj(x)
            x_cls, x = x[:1], x[1:]     
            tok, B, dim = x.shape
            H = int(tok**0.5)
            x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

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
            x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]

            x = torch.cat([x_cls, x], dim=0)
            return x
        else:
            B, dim, H, _ = x.shape
            # [8,1024,7,7]

            x_loc1 = self.conv_adapter_layers1(x)
            x_loc2 = self.conv_adapter_layers2(x)
            x_loc = torch.cat([x_loc1, x_loc2], dim=1)
            avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
            max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
            agg = torch.cat([avg_x, max_x], dim=1)
            y = self.conv_adapter_layers(agg)
            # y = y.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]
            y = F.sigmoid(y)
            x = x_loc1 * y[:, 0].unsqueeze(1) + x_loc2 * y[:, 1].unsqueeze(1)
            x = self.conv_adapter_final(x)

            return x
    
    def forward_proj(self, x):
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

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
        x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]
        return x






class Conv_Adapter_v8(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        ECA_combined, max and min use the same conv
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        layers1 = [
            nn.Conv2d(in_dim, 
                      in_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                    #   groups=in_dim,
                      bias=False),
            nn.BatchNorm2d(in_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      in_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                    #   groups=in_dim,
                      bias=False),
            nn.BatchNorm2d(in_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)

        layers = [
            nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        ]

        self.conv_adapter_layers = nn.Sequential(*layers)

    def forward(self, x, proj=False):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        avg_x = torch.mean(x_loc, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc, dim=1, keepdim=True)
        avg_x = self.conv_adapter_layers(avg_x)
        max_x = self.conv_adapter_layers(max_x)
        avg_x = F.sigmoid(avg_x)
        max_x = F.sigmoid(max_x)

        x_loc1 = self.conv_adapter_layers1(x_loc)
        x_loc2 = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1 * max_x
        x_loc2 = x_loc2 * avg_x
        x = x_loc1 + x_loc2
        
        x = x.reshape(B, -1, tok).permute(2, 0, 1)
        x = torch.cat([x_cls, x], dim=0)
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
    


class Conv_Adapter_v9(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3, text_dim=512):
        """
        text-guided, 用textfusion之后的特征取max、avg, 与textfusion之后的特征相乘
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        hidden_dim = text_dim   # in_dim//2
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(text_dim)
        self.adapter_norm_q2 = nn.LayerNorm(text_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.scale = text_dim ** -0.5
        self.adapter_cattn = CrossAttention_woproj(text_dim)

    # @get_local('agg')
    def forward(self, x, proj=False, text_fea=None, channel_first=True):
        if channel_first:
            x_cls, x = x[:1], x[1:]     
            tok, B, dim = x.shape
            H = int(tok**0.5)
            x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

            x_loc1 = self.conv_adapter_layers1(x_loc)
            x_loc2 = self.conv_adapter_layers2(x_loc)
            x_loc1 = x_loc1.reshape(B, -1, tok)
            x_loc2 = x_loc2.reshape(B, -1, tok)

            # x_loc1 = self.adapter_norm_q1(x_loc1)
            # x_loc2 = self.adapter_norm_q2(x_loc2)
            # text_fea = self.adapter_norm_kv(text_fea)
            # attn1 = x_loc1.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
            # attn1 = attn1.softmax(dim=-1)
            # x_loc1 = attn1 @ text_fea.permute(1, 0, 2)
            # attn2 = x_loc2.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
            # attn2 = attn2.softmax(dim=-1)
            # x_loc2 = attn2 @ text_fea.permute(1, 0, 2)

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
            x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]

            x = torch.cat([x_cls, x], dim=0)
            return x
            # return x, y
        else:
            B, dim, H, _ = x.shape
            # [8,1024,7,7]
            x_loc1 = self.conv_adapter_layers1(x)
            x_loc2 = self.conv_adapter_layers2(x)
            x_loc1 = x_loc1.reshape(B, dim, -1)
            x_loc2 = x_loc2.reshape(B, dim, -1)

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

            return x
            # return x, y
        



class Conv_Adapter_v10(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        text-guided, 用textfusion之后的特征取max、avg, 与textfusion之前的特征相乘
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        text_dim = 512
        hidden_dim = text_dim   # in_dim//2
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(text_dim)
        self.adapter_norm_q2 = nn.LayerNorm(text_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.scale = text_dim ** -0.5
        self.adapter_cattn = CrossAttention_woproj(text_dim)

    def forward(self, x, proj=False, text_fea=None):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x_loc1_bef = self.conv_adapter_layers1(x_loc)
        x_loc2_bef = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1_bef.reshape(B, -1, tok)
        x_loc2 = x_loc2_bef.reshape(B, -1, tok)

        # x_loc1 = self.adapter_norm_q1(x_loc1)
        # x_loc2 = self.adapter_norm_q2(x_loc2)
        # text_fea = self.adapter_norm_kv(text_fea)
        # attn1 = x_loc1.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
        # attn1 = attn1.softmax(dim=-1)
        # x_loc1 = attn1 @ text_fea.permute(1, 0, 2)
        # attn2 = x_loc2.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # x_loc2 = attn2 @ text_fea.permute(1, 0, 2)

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
        x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]

        x = torch.cat([x_cls, x], dim=0)
        return x



class CrossAttention(nn.Module):
    def __init__(self, in_dim, out_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.num_heads = num_heads
        # head_dim = dim // num_heads
        head_dim = out_dim
        self.scale = qk_scale or head_dim ** -0.5

        self.attn_drop = nn.Dropout(attn_drop)
        # self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q = nn.Linear(in_dim, out_dim, bias=qkv_bias)
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
    

class Conv_Adapter_v11(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        text-guided, 用textfusion之后的特征取max、avg, 与textfusion之后的特征相乘
        cattn有可学习参数
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        text_dim = 512
        hidden_dim = text_dim   # in_dim//2
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(text_dim)
        self.adapter_norm_q2 = nn.LayerNorm(text_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.scale = text_dim ** -0.5
        self.adapter_cattn1 = CrossAttention(text_dim)
        self.adapter_cattn2 = CrossAttention(text_dim)

    def forward(self, x, proj=False, text_fea=None):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x_loc1 = self.conv_adapter_layers1(x_loc)
        x_loc2 = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1.reshape(B, -1, tok)
        x_loc2 = x_loc2.reshape(B, -1, tok)

        # x_loc1 = self.adapter_norm_q1(x_loc1)
        # x_loc2 = self.adapter_norm_q2(x_loc2)
        # text_fea = self.adapter_norm_kv(text_fea)
        # attn1 = x_loc1.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
        # attn1 = attn1.softmax(dim=-1)
        # x_loc1 = attn1 @ text_fea.permute(1, 0, 2)
        # attn2 = x_loc2.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # x_loc2 = attn2 @ text_fea.permute(1, 0, 2)

        x_loc1 = x_loc1 + self.adapter_cattn1(
            self.adapter_norm_q1(x_loc1.permute(0, 2, 1)), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        ).permute(0, 2, 1)
        x_loc2 = x_loc2 + self.adapter_cattn2(
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
        x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]

        x = torch.cat([x_cls, x], dim=0)
        return x





class Conv_Adapter_v12(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        text-guided, 用textfusion之后的特征取max、avg, 与textfusion之后的特征相乘
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        text_dim = 512
        hidden_dim = text_dim   # in_dim//2
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=3, 
                      stride=stride, 
                      padding=3//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(text_dim)
        self.adapter_norm_q2 = nn.LayerNorm(text_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.scale = text_dim ** -0.5
        self.adapter_cattn = CrossAttention_woproj(text_dim)

    def forward(self, x, proj=False, text_fea=None):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)
        x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

        x_loc1 = self.conv_adapter_layers1(x_loc)
        x_loc2 = self.conv_adapter_layers2(x_loc)
        x_loc1 = x_loc1.reshape(B, -1, tok)
        x_loc2 = x_loc2.reshape(B, -1, tok)

        # x_loc1 = self.adapter_norm_q1(x_loc1)
        # x_loc2 = self.adapter_norm_q2(x_loc2)
        # text_fea = self.adapter_norm_kv(text_fea)
        # attn1 = x_loc1.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
        # attn1 = attn1.softmax(dim=-1)
        # x_loc1 = attn1 @ text_fea.permute(1, 0, 2)
        # attn2 = x_loc2.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # x_loc2 = attn2 @ text_fea.permute(1, 0, 2)

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

        # x_loc = torch.cat([x_loc1, x_loc2], dim=1)
        avg_x = torch.mean(x_loc1, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc2, dim=1, keepdim=True)
        agg = torch.cat([avg_x, max_x], dim=1)
        y = self.conv_adapter_layers(agg)
        # y = y.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]
        y = F.sigmoid(y)
        x = x_loc1 * y[:, 0].unsqueeze(1) + x_loc2 * y[:, 1].unsqueeze(1)
        x = self.conv_adapter_final(x)
        x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]

        x = torch.cat([x_cls, x], dim=0)
        return x




class Conv_Adapter_v13(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3):
        """
        ECA_combined, max and min use the same conv
        Text-guided
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        layers = [
            nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        ]

        self.conv_adapter_layers = nn.Sequential(*layers)

        text_dim = 512
        hidden_dim = text_dim
        self.adapter_norm_q1 = nn.LayerNorm(in_dim)
        self.adapter_norm_q2 = nn.LayerNorm(in_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.adapter_cattn1 = CrossAttention(in_dim, text_dim)
        self.adapter_cattn2 = CrossAttention(in_dim, text_dim)

        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

    def forward(self, x, proj=False, text_fea=None):
        x_cls, x = x[:1], x[1:]     
        tok, B, dim = x.shape
        H = int(tok**0.5)

        x_loc = x.permute(1, 0, 2)
        x_loc1 = self.adapter_cattn1(
            self.adapter_norm_q1(x_loc), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        )
        x_loc2 = self.adapter_cattn2(
            self.adapter_norm_q2(x_loc), 
            self.adapter_norm_kv(text_fea.permute(1, 0, 2))
        )

        x_loc1 = x_loc1.permute(0, 2, 1).reshape(B, -1, H, H)
        x_loc2 = x_loc2.permute(0, 2, 1).reshape(B, -1, H, H)

        avg_x = torch.mean(x_loc1, dim=1, keepdim=True)  # [B, 1, H, W]
        max_x, _ = torch.max(x_loc2, dim=1, keepdim=True)
        avg_x = self.conv_adapter_layers(avg_x)
        max_x = self.conv_adapter_layers(max_x)
        x_loc1 = x_loc1 * F.sigmoid(avg_x)
        x_loc2 = x_loc2 * F.sigmoid(max_x)

        # y = avg_x + max_x
        # y = y.reshape(B, -1, tok).permute(2, 0, 1)
        # y = F.sigmoid(y)
        # x = x * y.expand_as(x)
        x_loc = x_loc1 + x_loc2
        x_loc = self.conv_adapter_final(x_loc)
        x = x_loc.reshape(B, -1, tok).permute(2, 0, 1)
        
        x = torch.cat([x_cls, x], dim=0)
        return x
    

class Conv_Adapter_v14(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, hidden_dim=None, act='hs', kernel_size=3, text_dim=512):
        """
        v9: conv2用 1x1版本, 减少参数量
        text-guided, 用textfusion之后的特征取max、avg, 与textfusion之后的特征相乘
        :param in_dim: the input dimension
        :param out_dim: the output dimension. The input and output dimension should be the same.
        :param stride: stride of the depth-wise convolution.
        :param expand_ratio: expansion ratio of the hidden dimension.
        :param act: the activation function.
                    relu: ReLU
                    hs: h_swish
        """
        super().__init__()
        hidden_dim = text_dim   # in_dim//2
        layers1 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers1 = nn.Sequential(*layers1)
        layers2 = [
            nn.Conv2d(in_dim, 
                      hidden_dim, 
                      kernel_size=1, 
                      stride=stride, 
                      padding=1//2,  
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            # h_swish() if act=='hs' else nn.ReLU6(inplace=True),
            nn.GELU()
        ]
        self.conv_adapter_layers2 = nn.Sequential(*layers2)
        self.conv_adapter_layers = nn.Conv2d(2, 2, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.conv_adapter_final = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=stride, padding=1//2, bias=False)

        self.adapter_norm_q1 = nn.LayerNorm(text_dim)
        self.adapter_norm_q2 = nn.LayerNorm(text_dim)
        self.adapter_norm_kv = nn.LayerNorm(text_dim)
        self.scale = text_dim ** -0.5
        self.adapter_cattn = CrossAttention_woproj(text_dim)

    def forward(self, x, proj=False, text_fea=None, channel_first=True):
        if channel_first:
            x_cls, x = x[:1], x[1:]     
            tok, B, dim = x.shape
            H = int(tok**0.5)
            x_loc = x.permute(1, 2, 0).reshape(B, dim, H, H)

            x_loc1 = self.conv_adapter_layers1(x_loc)
            x_loc2 = self.conv_adapter_layers2(x_loc)
            x_loc1 = x_loc1.reshape(B, -1, tok)
            x_loc2 = x_loc2.reshape(B, -1, tok)

            # x_loc1 = self.adapter_norm_q1(x_loc1)
            # x_loc2 = self.adapter_norm_q2(x_loc2)
            # text_fea = self.adapter_norm_kv(text_fea)
            # attn1 = x_loc1.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
            # attn1 = attn1.softmax(dim=-1)
            # x_loc1 = attn1 @ text_fea.permute(1, 0, 2)
            # attn2 = x_loc2.permute(0, 2, 1) @ text_fea.permute(1, 2, 0) * self.scale
            # attn2 = attn2.softmax(dim=-1)
            # x_loc2 = attn2 @ text_fea.permute(1, 0, 2)

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
            x = x.reshape(B, -1, tok).permute(2, 0, 1)  # [tok, B, 2]

            x = torch.cat([x_cls, x], dim=0)
            return x
        else:
            B, dim, H, _ = x.shape
            # [8,1024,7,7]
            x_loc1 = self.conv_adapter_layers1(x)
            x_loc2 = self.conv_adapter_layers2(x)
            x_loc1 = x_loc1.reshape(B, dim, -1)
            x_loc2 = x_loc2.reshape(B, dim, -1)

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

            return x



class ECA_combined(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_y = self.avgpool(x)
        avg_y = self.conv(avg_y.permute(0, 2, 1))
        max_y = self.maxpool(x)
        max_y = self.conv(max_y.permute(0, 2, 1))
        y = avg_y + max_y
        y = y.permute(0, 2, 1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
    

