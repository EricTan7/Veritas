import os

# from swift import get_logger
# from swift.llm import TrainArguments, ModelArch, TemplateType, Model, ModelGroup, ModelMeta, register_model, ModelInfo, \
#     register_template, Template, register_model_arch, MultiModelKeys
# from swift.llm.model.model.internlm import get_model_tokenizer_internvl
# from swift.llm.template.template.internvl import InternvlTemplate
# from swift.llm.template.template.utils import ChatmlTemplateMeta
# from swift.llm.template.template_inputs import StdTemplateInputs
# from swift.llm.template.utils import Context, findall
# from swift.llm.template.vision_utils import transform_image
# from swift.llm.train import SwiftSft

# import torch
# from torch import nn
# from typing import List, Union, Literal
# from typing import Any, Dict
# from PIL import Image
# from swift.utils import get_model_parameter_info, get_env_args

# from reg_class import CustomizedInternvlTemplate, get_model_tokenizer_customizedinternvl

# try:
#     import orjson as json
# except:
#     import json

# from model.internvl_chat import (InternVLChatConfig,CustomizedInternVLChatModel)

# from transformers import AutoTokenizer

# from myswift.Trainer import CustomizedSeq2SeqTrainer


from functools import partial
from typing import Any, Dict

from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification,
                          AutoTokenizer, GenerationConfig, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerBase)

from swift.llm import TemplateType
from ..constant import LLMModelType, MLLMModelType, RMModelType
from ..model_arch import ModelArch
from ..patcher import patch_output_clone, patch_output_to_input_device
from ..register import (Model, ModelGroup, ModelMeta, register_model)
from ..utils import ModelInfo, safe_snapshot_download, use_submodel_func
from ..utils import AttnImpl, HfConfigFactory, ModelInfo, safe_snapshot_download

from .internvl_chat import InternVLChatModel_Custom


logger = get_logger()



def get_model_tokenizer_from_local(model_dir: str,
                                   model_info: ModelInfo,
                                   model_kwargs: Dict[str, Any],
                                   load_model: bool = True,
                                   *,
                                   tokenizer=None,
                                   model_config=None,
                                   automodel_class=None,
                                   **kwargs):
    """Load the model and tokenizer from the local model_dir."""
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    # fix prediction_step (internvl2, ovis, ...)
    if not hasattr(model_config, 'keys_to_ignore_at_inference'):
        model_config.keys_to_ignore_at_inference = []
    if 'past_key_values' not in model_config.keys_to_ignore_at_inference:
        model_config.keys_to_ignore_at_inference.append('past_key_values')

    torch_dtype = model_info.torch_dtype
    model_config.torch_dtype = torch_dtype
    HfConfigFactory.compat_zero3(model_config)
    rope_scaling = kwargs.get('rope_scaling')
    if rope_scaling:
        HfConfigFactory.set_config_attr(model_config, 'rope_scaling', rope_scaling)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    num_labels = model_info.num_labels or getattr(model_config, 'num_labels', None)
    if num_labels and model_info.task_type == 'seq_cls':
        model_info.num_labels = num_labels
        model_config.num_labels = num_labels

    model = None
    if load_model:
        # _patch_awq_compat(model_info)     # No need for quant (custom model)
        logger.info(f'model_kwargs: {model_kwargs}')
        # fix seq_cls
        if model_info.task_type == 'seq_cls' and automodel_class is None:
            try:
                model = InternVLChatModel_Custom.from_pretrained(
                    model_dir, config=model_config, torch_dtype=torch_dtype, trust_remote_code=True, **model_kwargs)
            except ValueError:
                model = None

        automodel_class = automodel_class or AutoModelForCausalLM
        model_meta = kwargs['model_meta']

        # fix not save modeling_xxx.py (transformers 4.45)
        # https://github.com/huggingface/transformers/issues/24737
        has_remote_code = hasattr(model_config, 'auto_map') and automodel_class.__name__ in model_config.auto_map
        if has_remote_code and model._auto_class is None:
            model._auto_class = automodel_class.__name__

        if model_info.task_type == 'embedding' and automodel_class.__name__ != 'AutoModel':
            from swift.llm.model.patcher import patch_output_normalizer
            patch_output_normalizer(model, model_meta=model_meta)

    model_info.config = model_config if model is None else model.config
    if model:
        # fix seq classification task
        pad_token_id = model.config.pad_token_id or tokenizer.pad_token_id
        HfConfigFactory.set_model_config_attr(model, 'pad_token_id', pad_token_id)
    return model, tokenizer



def get_model_tokenizer_with_flash_attn(model_dir: str,
                                        model_info: ModelInfo,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    model_config = kwargs.get('model_config')
    if model_config is None:
        model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    AttnImpl.update_attn_impl(model_config, kwargs.get('attn_impl'), kwargs.get('attn_impl_keys'))
    kwargs['model_config'] = model_config
    return get_model_tokenizer_from_local(model_dir, model_info, model_kwargs, load_model, **kwargs)



def get_model_tokenizer_internvl_custom(model_dir: str,
                                        model_info: ModelInfo,
                                        model_kwargs: Dict[str, Any],
                                        load_model: bool = True,
                                        **kwargs):
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, model_info, model_kwargs, load_model, **kwargs)

    if model_info.quant_method == 'bnb' and kwargs.get('is_training'):
        # patch: bnb backward shape mismatch bug
        if model is not None and model.language_model is not None:
            model.language_model.output.state.force_no_igemmlt = True

    if model is not None:
        use_submodel_func(model, 'language_model')
        patch_output_clone(model.language_model.get_input_embeddings())

    return model, tokenizer


