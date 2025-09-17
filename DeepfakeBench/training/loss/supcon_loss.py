from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False



def supcon_loss(logits, mask, logits_mask=None, ignore=None):
    """
        Supervised Contrastive Loss
        Args:
            logits: predicted logits [B, C] (C: number of samples from the second mdality)
            mask: ground truth label to perform supervised masking [B, C]
            logits_mask: ignoring some contrastive pairs [B, C]
        Returns:
            loss_cl: contrastive loss
    """
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    exp_mixed_logits = torch.exp(logits)
    if logits_mask is not None:
        exp_mixed_logits = exp_mixed_logits * logits_mask

    log_prob = logits - torch.log(exp_mixed_logits.sum(1, keepdim=True))  # cross entropy + softmax, partition the log
    num_pos_pairs = mask.sum(1)
    num_pos_pairs = torch.where(num_pos_pairs < 1e-6, 1, num_pos_pairs) # numerical stability
    mean_log_prob_pos = (mask * log_prob).sum(1) / num_pos_pairs    # sum over positive pairs, division is outside the log
    
    # loss_cl = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    loss_cl = -mean_log_prob_pos
    if ignore is not None:
        loss_cl[ignore] = 0
    loss_cl = loss_cl.mean()     # mean over batch samples

    return loss_cl



def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
    else:
        gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_image_features[rank] = image_features
            gathered_text_features[rank] = text_features
        all_image_features = torch.cat(gathered_image_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


def gather_features_single(
        features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    # We gather tensors from all gpus
    if gather_with_grad:
        all_features = torch.cat(torch.distributed.nn.all_gather(features), dim=0)
    else:
        gathered_features = [torch.zeros_like(features) for _ in range(world_size)]
        dist.all_gather(gathered_features, features)
        if not local_loss:
            # ensure grads for local rank when all_* features don't have a gradient
            gathered_features[rank] = features
        all_features = torch.cat(gathered_features, dim=0)

    return all_features



class SupConLoss(nn.Module):
    def __init__(
            self,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def get_logits(self, image_features, text_features, logit_scale, dual_loss):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features,
                text_features,
                local_loss=False,
                gather_with_grad=False,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            logits_per_image = logit_scale * all_image_features @ all_text_features.T
            logits_per_text = logits_per_image.T if dual_loss else None
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T if dual_loss else None
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, labels, dual_loss=False):
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale, dual_loss)

        if dual_loss:
            total_loss = (
                supcon_loss(logits_per_image, labels) +
                supcon_loss(logits_per_text, labels)
            ) / 2
        else:
            total_loss = supcon_loss(logits_per_image, labels)

        return total_loss



class SupConLoss_Single(nn.Module):
    def __init__(
            self,
            rank=0,
            world_size=1,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

    def get_logits(self, features, logit_scale):
        if self.world_size > 1:
            all_features = gather_features_single(
                features,
                local_loss=False,
                gather_with_grad=False,
                rank=self.rank,
                world_size=self.world_size,
                use_horovod=self.use_horovod,
            )
            logits = logit_scale * features @ features.T
        else:
            logits = logit_scale * features @ features.T
        
        return logits

    def forward(self, features, logit_scale, labels):
        logits_per_image = self.get_logits(features, logit_scale)
        total_loss = supcon_loss(logits_per_image, labels)

        return total_loss

