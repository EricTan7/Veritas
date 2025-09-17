import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import os
import re
import torch.nn.functional as F
from timm.models.layers import DropPath, Mlp
import networks.clip as clip
from utils.model_utils import convert_params_to_value
from networks.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from .basic import shuffle_unit, weights_init_classifier, weights_init_kaiming, \
    TextEncoder, CrossAttention, Gated_CrossAttention, \
    CrossAttnBlock, CoAttnBlock, Gated_CoAttnBlock, SoftmaxWithTemperature, \
    LNLogitScale, FocusResampler, GatedCrossAttnBlock, LNLogitScaleCSC, \
    LNLogitScaleCSC2, TextEncoder_remoteclip
# from visualizer import get_local
# get_local.activate()
from peft import LoraConfig, get_peft_model
from transformers import BertTokenizer, BertModel
from copy import deepcopy
np.set_printoptions(threshold=np.inf)
import ipdb

_tokenizer = _Tokenizer()
device = 'cuda'




class FreqEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        # 第一层卷积：7x7卷积层 + BatchNorm + ReLU
        self.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=64, kernel_size=7, stride=1, padding=3)  # 64输出通道
        self.bn1 = nn.BatchNorm2d(64)
        
        # 最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 第二层卷积：1x1卷积 + BatchNorm + ReLU
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(128)

        # 全连接层
        # self.fc1 = nn.Linear(128 * 128 * 128, 1024)  # 假设经过池化后尺寸为128x128
        # self.bn3 = nn.BatchNorm1d(1024)
        
        # 第三层卷积：1x1卷积 + BatchNorm + ReLU
        self.conv3 = nn.Conv2d(128, out_dim, kernel_size=1, stride=1)
        self.bn4 = nn.BatchNorm2d(out_dim)
        
        # # 特征维度
        # self.fc3 = nn.Linear(1024, 256)

    def forward(self, x):
        # 第一层卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 最大池化
        x = self.pool(x)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # 第三层卷积
        x = self.conv3(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # Flatten
        
        return x





class BasePrompter(nn.Module):
    """
        Base Class for Textual Prompter
    """
    def __init__(self):
        super().__init__()

    def freeze(self, config):
        transfer_type = config["model_config"]["text_transfer_type"]
        tuned_layers = convert_params_to_value(config["model_config"]["text_tuned_layers"]) if config["model_config"].get("text_tuned_layers", False) else [-1]
        if transfer_type == "no_freeze":
            pass
        elif transfer_type in ["freeze_all", "VPT"]:
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = True if "VPT" in name else False
        elif transfer_type == "Adapter":
            for name, param in self.text_encoder.named_parameters():
                param.requires_grad = True if "adapter" in name or "VPT" in name else False
        elif transfer_type in ['lora', 'lora_vpt']:
            self.text_encoder.transformer.attn_replace()
            all_lora_layers = list()
            for n, m in self.text_encoder.named_modules():
                # find current layer num
                layer = re.findall(r'\d+', n)
                layer = -1 if len(layer)==0 else int(layer[0])
                if 'q_proj' in n and 'q_proj' in config["model_config"]["lora_config"]["text_lora_layers"] and layer in tuned_layers:
                    all_lora_layers.append(n)
                if 'k_proj' in n and 'k_proj' in config["model_config"]["lora_config"]["text_lora_layers"] and layer in tuned_layers:
                    all_lora_layers.append(n)
                if 'v_proj' in n and 'v_proj' in config["model_config"]["lora_config"]["text_lora_layers"] and layer in tuned_layers:
                    all_lora_layers.append(n)
                if 'out_proj' in n and 'out_proj' in config["model_config"]["lora_config"]["text_lora_layers"] and layer in tuned_layers:
                    all_lora_layers.append(n)
                if 'mlp' in config["model_config"]["lora_config"]["text_lora_layers"] and layer in tuned_layers:
                    if 'c_fc' in n or 'c_proj' in n:
                        all_lora_layers.append(n)
            loraconfig = LoraConfig(
                r=config["model_config"]["lora_config"]["r"],
                lora_alpha=config["model_config"]["lora_config"]["alpha"],
                lora_dropout=config["model_config"]["lora_config"]["dropout"],
                bias="none",
                target_modules=all_lora_layers,
            )
            self.text_encoder = get_peft_model(self.text_encoder, loraconfig)
            for name, param in self.text_encoder.named_parameters():
                if "VPT" in name:
                    param.requires_grad = True
        else:
            raise NotImplementedError


    def forward(self, image_fea):
        pass
 


class Prompter(BasePrompter):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config["model_config"]["n_ctx_text"]
        ctx_init = config["model_config"]["ctx_init"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init and not config["model_config"]["prompt_csc"]:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
            prompt_prefix = ctx_init

        else:
            # random initialization
            if config["model_config"]["prompt_csc"]:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                if ctx_init:
                    ctx_init = ctx_init.replace("_", " ")
                    n_ctx = len(ctx_init.split(" "))
                    prompt = clip.tokenize(ctx_init)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompt).type(dtype)
                    ctx_vectors[:] = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
                    prompt_prefix = ctx_init
                else:
                    nn.init.normal_(ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
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

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        # NOTE: the following is not "Parameters", (1) will not be saved when in save_model(), (2) do not have grads
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.text_encoder = TextEncoder(clip_model)
        self.freeze(config)

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim),
                # padding, # (n_cls, 1, dim)
            ],
            dim=1,
        )
        tokenized_prompts = self.tokenized_prompts

        prompt_features = self.text_encoder(prompts, tokenized_prompts)

        return prompt_features
 



class AsyConditionedPrompter(BasePrompter):
    """
        Assymetrically conditioned prompter
            Real: X X X X X X X X real image
            Fake: X X X X X X X X [IMAGE] fake image 
        Alternatively, can remove texts (only keep learnable prompts)
            Real: X X X X X X X X
            Fake: X X X X X X X X [IMAGE]
    """
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config["model_config"]["n_ctx_text"]
        ctx_init = config["model_config"]["ctx_init"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init and not config["model_config"]["prompt_csc"]:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
            prompt_prefix = ctx_init

        else:
            # random initialization
            if config["model_config"]["prompt_csc"]:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                if ctx_init:
                    ctx_init = ctx_init.replace("_", " ")
                    n_ctx = len(ctx_init.split(" "))
                    prompt = clip.tokenize(ctx_init)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompt).type(dtype)
                    ctx_vectors[:] = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
                    prompt_prefix = ctx_init
                else:
                    nn.init.normal_(ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
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
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        if config['model_config'].get('prompt_with_class', False) and config['model_config']['prompt_with_class']:
            prompts_real = prompt_prefix + " " + classnames[0] + "."
            prompts_fake = prompt_prefix + " " + "image" + " " + classnames[1] + "."    # placeholder for image embedding
        else:
            prompts_real = prompt_prefix
            prompts_fake = prompt_prefix + " " + "image"   # placeholder for image embedding
        prompts = [prompts_real, prompts_fake]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # tokenized_prompts_real = clip.tokenize(prompts[0])
        # tokenized_prompts_fake = clip.tokenize(prompts[1])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_real", embedding[:1, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("token_suffix_fake", embedding[1:, 2 + n_ctx :, :])  # CLS, EOS

        # NOTE: the following is not "Parameters", (1) will not be saved when in save_model(), (2) do not have grads
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.text_encoder = TextEncoder(clip_model)
        self.freeze(config)

    def forward(self, image_fea):
        if len(self.ctx.shape) == 3:
            ctx_real, ctx_fake = self.ctx[:1], self.ctx[1:]
        else:
            ctx_real, ctx_fake = self.ctx.unsqueeze(0), self.ctx.unsqueeze(0)
        B, _ = image_fea.shape
        image_fea = image_fea.unsqueeze(1)

        prefix_real, prefix_fake = self.token_prefix[:1], self.token_prefix[1:]
        suffix_real, suffix_fake = self.token_suffix_real, self.token_suffix_fake    # placeholder for image embedding
        prefix_fake = prefix_fake.expand(B, -1, -1)
        suffix_fake = suffix_fake.expand(B, -1, -1)
        ctx_fake = ctx_fake.expand(B, -1, -1)

        # ipdb.set_trace()
        prompts_real = torch.cat([prefix_real, ctx_real, suffix_real], dim=1)    # [1, 77, dim]
        prompts_fake = torch.cat([prefix_fake, ctx_fake, image_fea, suffix_fake], dim=1)     # [1,77,dim]
        # prompts_fake = torch.cat([prefix_fake, image_fea, ctx_fake, suffix_fake], dim=1)     # [1,77,dim]
        prompts = torch.cat([prompts_real, prompts_fake], dim=0)

        tokenized_prompts_real = self.tokenized_prompts[:1]
        tokenized_prompts_fake = self.tokenized_prompts[1:]
        tokenized_prompts_fake = tokenized_prompts_fake.expand(B, -1)
        tokenized_prompts = torch.cat([tokenized_prompts_real, tokenized_prompts_fake], dim=0)
        # print(self.tokenized_prompts.shape, tokenized_prompts.shape)    # [1+B, 77]

        prompt_features = self.text_encoder(prompts, tokenized_prompts)
        # print(prompt_features.shape)    # [1+B, dim]

        return prompt_features
 


class ConditionedPrompter(BasePrompter):
    """
        Conditioned prompter
        Real: a photo of a [image] real iamge
        Fake: a photo of a [image] fake image 
    """
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config["model_config"]["n_ctx_text"]
        ctx_init = config["model_config"]["ctx_init"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init and not config["model_config"]["prompt_csc"]:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
            prompt_prefix = ctx_init

        else:
            # random initialization
            if config["model_config"]["prompt_csc"]:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                if ctx_init:
                    ctx_init = ctx_init.replace("_", " ")
                    n_ctx = len(ctx_init.split(" "))
                    prompt = clip.tokenize(ctx_init)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompt).type(dtype)
                    ctx_vectors[:] = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
                    prompt_prefix = ctx_init
                else:
                    nn.init.normal_(ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
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
        prompts = [prompt_prefix + " " + "image" + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 2 + n_ctx :, :])  # CLS, EOS

        # NOTE: the following is not "Parameters", (1) will not be saved when in save_model(), (2) do not have grads
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.text_encoder = TextEncoder(clip_model)
        self.freeze(config)

    def forward(self, image_fea):
        ctx_real, ctx_fake = self.ctx[:1], self.ctx[1:]
        B, _ = image_fea.shape
        image_fea = image_fea.unsqueeze(1)

        prefix_real, prefix_fake = self.token_prefix[:1], self.token_prefix[1:]
        suffix_real, suffix_fake = self.token_suffix[:1], self.token_suffix[1:]    # placeholder for image embedding
        prefix_fake = prefix_fake.expand(B, -1, -1)
        prefix_real = prefix_real.expand(B, -1, -1)
        suffix_fake = suffix_fake.expand(B, -1, -1)
        suffix_real = suffix_real.expand(B, -1, -1)
        ctx_real = ctx_real.expand(B, -1, -1)
        ctx_fake = ctx_fake.expand(B, -1, -1)

        # ipdb.set_trace()

        prompts_real = torch.cat([prefix_real, ctx_real, image_fea, suffix_real], dim=1)    # [1, 77, dim]
        prompts_fake = torch.cat([prefix_fake, ctx_fake, image_fea, suffix_fake], dim=1)     # [1,77,dim]
        prompts = torch.cat([prompts_real, prompts_fake], dim=0)

        tokenized_prompts_real = self.tokenized_prompts[:1]
        tokenized_prompts_fake = self.tokenized_prompts[1:]
        tokenized_prompts_real = tokenized_prompts_real.expand(B, -1)
        tokenized_prompts_fake = tokenized_prompts_fake.expand(B, -1)
        tokenized_prompts = torch.cat([tokenized_prompts_real, tokenized_prompts_fake], dim=0)
        # print(self.tokenized_prompts.shape, tokenized_prompts.shape)    # [1+B, 77]

        prompt_features = self.text_encoder(prompts, tokenized_prompts)
        # print(prompt_features.shape)    # [1+B, dim]

        return prompt_features
    



class AsyConditionedPrompter_ZS(BasePrompter):
    """
        Assymetrically conditioned prompter
            Real: a real photo of a image
            Fake: a fake photo of a [IMAGE] image 
    """
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        n_ctx = 6
        templates = ["a real photo of a image", "a fake photo of a image image"]        # the second last `image` is a placeholder
        tokenized_prompts = clip.tokenize(templates)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("prompts_real", embedding[:1, :, :])
        self.register_buffer("token_prefix_fake", embedding[1:, :6, :])  # SOS
        self.register_buffer("token_suffix_fake", embedding[1:, 7:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        self.text_encoder = TextEncoder(clip_model)
        self.freeze(config)

    def forward(self, image_fea):
        B, _ = image_fea.shape
        image_fea = image_fea.unsqueeze(1)

        prefix_fake = self.token_prefix_fake
        suffix_fake = self.token_suffix_fake    # placeholder for image embedding
        prefix_fake = prefix_fake.expand(B, -1, -1)
        suffix_fake = suffix_fake.expand(B, -1, -1)

        prompts_real = self.prompts_real
        prompts_fake = torch.cat([prefix_fake, image_fea, suffix_fake], dim=1)     # [1,77,dim]
        prompts = torch.cat([prompts_real, prompts_fake], dim=0)

        tokenized_prompts_real = self.tokenized_prompts[:1]
        tokenized_prompts_fake = self.tokenized_prompts[1:]
        tokenized_prompts_fake = tokenized_prompts_fake.expand(B, -1)
        tokenized_prompts = torch.cat([tokenized_prompts_real, tokenized_prompts_fake], dim=0)
        # print(self.tokenized_prompts.shape, tokenized_prompts.shape)    # [1+B, 77]

        prompt_features = self.text_encoder(prompts, tokenized_prompts)
        # print(prompt_features.shape)    # [1+B, dim]

        return prompt_features
 




class ConditionedPrompter_ZS(BasePrompter):
    """
        Assymetrically conditioned prompter
            Real: a real photo of a image
            Fake: a fake photo of a [IMAGE] image 
    """
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        dtype = clip_model.dtype
        n_ctx = 6
        templates = ["a real photo of a image image", "a fake photo of a image image"]        # the second last `image` is a placeholder
        tokenized_prompts = clip.tokenize(templates)

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix_real", embedding[:1, :6, :])  # SOS
        self.register_buffer("token_suffix_real", embedding[:1, 7:, :])  # CLS, EOS
        self.register_buffer("token_prefix_fake", embedding[1:, :6, :])  # SOS
        self.register_buffer("token_suffix_fake", embedding[1:, 7:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        self.text_encoder = TextEncoder(clip_model)
        self.freeze(config)

    def forward(self, image_fea):
        B, _ = image_fea.shape
        image_fea = image_fea.unsqueeze(1)

        prefix_real, prefix_fake = self.token_prefix_real, self.token_prefix_fake
        suffix_real, suffix_fake = self.token_suffix_real, self.token_suffix_fake    # placeholder for image embedding
        prefix_fake = prefix_fake.expand(B, -1, -1)
        suffix_fake = suffix_fake.expand(B, -1, -1)
        prefix_real = prefix_real.expand(B, -1, -1)
        suffix_real = suffix_real.expand(B, -1, -1)

        prompts_real = torch.cat([prefix_real, image_fea, suffix_real], dim=1)     # [1,77,dim]
        prompts_fake = torch.cat([prefix_fake, image_fea, suffix_fake], dim=1)     # [1,77,dim]
        prompts = torch.cat([prompts_real, prompts_fake], dim=0)

        tokenized_prompts_real = self.tokenized_prompts[:1]
        tokenized_prompts_fake = self.tokenized_prompts[1:]
        tokenized_prompts_real = tokenized_prompts_real.expand(B, -1)
        tokenized_prompts_fake = tokenized_prompts_fake.expand(B, -1)
        tokenized_prompts = torch.cat([tokenized_prompts_real, tokenized_prompts_fake], dim=0)
        # print(self.tokenized_prompts.shape, tokenized_prompts.shape)    # [1+B, 77]

        prompt_features = self.text_encoder(prompts, tokenized_prompts)
        # print(prompt_features.shape)    # [1+B, dim]

        return prompt_features
 



class AsyConditionedPrompter_additive(BasePrompter):
    """
        Assymetrically conditioned prompter
            Real: X X X X X X X X real image
            Fake: X X X X X X X X fake image  (additive [IMAGE])
        Alternatively, can remove texts (only keep learnable prompts)
            Real: X X X X X X X X
            Fake: X X X X X X X X [IMAGE]
    """
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = config["model_config"]["n_ctx_text"]
        ctx_init = config["model_config"]["ctx_init"]
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        if ctx_init and not config["model_config"]["prompt_csc"]:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
            prompt_prefix = ctx_init

        else:
            # random initialization
            if config["model_config"]["prompt_csc"]:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
                if ctx_init:
                    ctx_init = ctx_init.replace("_", " ")
                    n_ctx = len(ctx_init.split(" "))
                    prompt = clip.tokenize(ctx_init)
                    with torch.no_grad():
                        embedding = clip_model.token_embedding(prompt).type(dtype)
                    ctx_vectors[:] = embedding[0, 1 : 1 + n_ctx, :]    # remove sot_token & eot_token
                    prompt_prefix = ctx_init
                else:
                    nn.init.normal_(ctx_vectors, std=0.02)
                    prompt_prefix = " ".join(["X"] * n_ctx)
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
        # prompts = [prompt_prefix + " " + name + "." for name in classnames]
        if config['model_config'].get('prompt_with_class', False) and config['model_config']['prompt_with_class']:
            prompts_real = prompt_prefix + " " + classnames[0] + "."
            prompts_fake = prompt_prefix + " " + classnames[1] + "."    # placeholder for image embedding
        else:
            prompts_real = prompt_prefix
            prompts_fake = prompt_prefix   # placeholder for image embedding
        prompts = [prompts_real, prompts_fake]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # tokenized_prompts_real = clip.tokenize(prompts[0])
        # tokenized_prompts_fake = clip.tokenize(prompts[1])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_real", embedding[:1, 1 + n_ctx :, :])  # CLS, EOS
        self.register_buffer("token_suffix_fake", embedding[1:, 1 + n_ctx :, :])  # CLS, EOS

        # NOTE: the following is not "Parameters", (1) will not be saved when in save_model(), (2) do not have grads
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        self.text_encoder = TextEncoder(clip_model)
        self.freeze(config)

    def forward(self, image_fea):
        if len(self.ctx.shape) == 3:
            ctx_real, ctx_fake = self.ctx[:1], self.ctx[1:]
        else:
            ctx_real, ctx_fake = self.ctx.unsqueeze(0), self.ctx.unsqueeze(0)
        B, _ = image_fea.shape
        image_fea = image_fea.unsqueeze(1)

        prefix_real, prefix_fake = self.token_prefix[:1], self.token_prefix[1:]
        suffix_real, suffix_fake = self.token_suffix_real, self.token_suffix_fake    # placeholder for image embedding
        prefix_fake = prefix_fake.expand(B, -1, -1)
        suffix_fake = suffix_fake.expand(B, -1, -1)
        ctx_fake = ctx_fake.expand(B, -1, -1)

        # ipdb.set_trace()
        prompts_real = torch.cat([prefix_real, ctx_real, suffix_real], dim=1)    # [1, 77, dim]
        ctx_fake = ctx_fake + image_fea
        prompts_fake = torch.cat([prefix_fake, ctx_fake, suffix_fake], dim=1)     # [1,77,dim]
        # prompts_fake = torch.cat([prefix_fake, image_fea, ctx_fake, suffix_fake], dim=1)     # [1,77,dim]
        prompts = torch.cat([prompts_real, prompts_fake], dim=0)

        tokenized_prompts_real = self.tokenized_prompts[:1]
        tokenized_prompts_fake = self.tokenized_prompts[1:]
        tokenized_prompts_fake = tokenized_prompts_fake.expand(B, -1)
        tokenized_prompts = torch.cat([tokenized_prompts_real, tokenized_prompts_fake], dim=0)
        # print(self.tokenized_prompts.shape, tokenized_prompts.shape)    # [1+B, 77]

        prompt_features = self.text_encoder(prompts, tokenized_prompts)
        # print(prompt_features.shape)    # [1+B, dim]

        return prompt_features
 

