import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import torch.optim as optim
from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR
from optimizor.Warmup import ConstantWarmupScheduler, LinearWarmupScheduler



def choose_optimizer(model, config):
    opt_name = config['optimizer']['type']
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
        return optimizer
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(
            params=model.parameters(),
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
        return optimizer
    elif opt_name == 'sam':
        optimizer = SAM(
            model.parameters(), 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    return optimizer


def choose_optimizer_split_prompt(model, config, lr_mult=1.):
    opt_name = config['optimizer']['type']

    prompt_params, other_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        elif 'ctx' in pname or 'VPT' in pname:
            prompt_params.append(p)
        else:
            other_params.append(p)

    print("prompt params:", len(prompt_params))
    print("other params:", len(other_params))

    # optimizer
    if opt_name == 'sgd':
        optimizer = optim.SGD(
            params=other_params,
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
            weight_decay=config['optimizer'][opt_name]['weight_decay']
        )
    elif opt_name == 'adam':
        optimizer = optim.Adam(
            params=other_params,
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(
            params=other_params,
            lr=config['optimizer'][opt_name]['lr'],
            weight_decay=config['optimizer'][opt_name]['weight_decay'],
            betas=(config['optimizer'][opt_name]['beta1'], config['optimizer'][opt_name]['beta2']),
            eps=config['optimizer'][opt_name]['eps'],
            amsgrad=config['optimizer'][opt_name]['amsgrad'],
        )
    elif opt_name == 'sam':
        optimizer = SAM(
            other_params, 
            optim.SGD, 
            lr=config['optimizer'][opt_name]['lr'],
            momentum=config['optimizer'][opt_name]['momentum'],
        )
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(config['optimizer']))
    
    # prompt optimizer
    if len(prompt_params) == 0:
        optimizer_prompt = None
    else:
        optimizer_prompt = optim.SGD(
            params=prompt_params,
            lr=config['optimizer']['sgd']['lr'],
            momentum=config['optimizer']['sgd']['momentum'],
            weight_decay=config['optimizer']['sgd']['weight_decay']
        )

    return optimizer, optimizer_prompt



def choose_scheduler(config, optimizer):
    if optimizer is None:
        return None
    # decay
    sched_name = config['lr_scheduler']['type']
    opt_name = config['optimizer']['type']
    max_epoch = config['lr_scheduler']['max_epoch']
    min_lr = config['lr_scheduler']['min_lr']
    max_epoch = max_epoch if max_epoch is not None else config['nEpochs']
    min_lr = min_lr if min_lr is not None else config['optimizer'][opt_name]['lr'] * 0.005

    if sched_name is None:
        return None
    elif sched_name == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['lr_step'],
            gamma=config['lr_gamma'],
        )
    elif sched_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epoch,
            eta_min=min_lr,
        )
    elif sched_name == 'linear':
        scheduler = LinearDecayLR(
            optimizer,
            n_epoch=config['nEpochs'],
            start_decay=int(config['nEpochs']/4),
        )
    else:
        raise NotImplementedError('Scheduler {} is not implemented'.format(sched_name))
    
    # warm up
    warmup_name = config['lr_scheduler']['warmup']['type']
    warmup_epoch = config['lr_scheduler']['warmup']['epoch']
    warmup_lr = config['optimizer'][opt_name]['lr'] * 0.1
    if warmup_name is None or warmup_name == 'no_warmup':
        return scheduler
    elif warmup_name == "constant":
        scheduler = ConstantWarmupScheduler(
            optimizer, scheduler, warmup_epoch, warmup_lr
        )
    elif warmup_name == "linear":
        scheduler = LinearWarmupScheduler(
            optimizer, scheduler, warmup_epoch, warmup_lr
        )
    else:
        raise NotImplementedError('Warmup {} is not implemented'.format(warmup_name))
    
    return scheduler
        