# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: training code.

import os
import argparse
import datetime
import yaml
from datetime import timedelta

import torch
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
import torch.distributed as dist

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset import prepare_training_data, prepare_testing_data, prepare_val_data
from optimizor import choose_optimizer, choose_scheduler, choose_optimizer_split_prompt

from utils.metrics import choose_metric, parse_metric_for_print
from utils.logger import create_logger, RankFilter
from utils.config import get_config
from utils.model_utils import set_seed, thread_flag, check_trainable_params, getModelSize
import wandb
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Process some paths.')
parser.add_argument('--detector_cfg', type=str,
                    default='./training/config/detector/xception.yaml',
                    help='path to detector YAML file')
parser.add_argument('--dataset_cfg', type=str,
                    default='./training/config/dataset/face_common.yaml',
                    help='path to dataset YAML file')
parser.add_argument("--train_dataset", nargs="+")
parser.add_argument("--val_dataset", nargs="+")
parser.add_argument("--test_dataset", nargs="+")
parser.add_argument("--ddp", action='store_true', default=False)
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--task_target', type=str, default="", help='specify the target of current training task')
parser.add_argument("--wandb_proj", help='project name of wandb', type=str, default="AIGCD")
parser.add_argument("--wandb_tags", help='tag name of wandb', type=str, default="dummy")
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def main():
    # config
    config = get_config(args)

    # init seed
    set_seed(config)
    
    # create logger
    timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    
    if config["data_aug"]["complex_aug"]:
        suffix = "_aug"
    else:
        suffix = ""
    if config['pretrained'].endswith(".pth"):
        model_prefix = config['pretrained'].split('/')[-1].split(".")[0]
    else:
        model_prefix = config['pretrained'].split('/')[-1]
    logger_path = os.path.join(
        config['log_dir'],
        model_prefix,
        config['model_config']['transfer_type'] + '_' + config['train_dataset'][0] + suffix,
        timenow
    )
    os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'training.log'))
    if config['ddp']:
        # dist.init_process_group(backend='gloo')
        dist.init_process_group(backend='nccl', timeout=timedelta(minutes=30))
        dist.barrier()
        logger.addFilter(RankFilter(0))

    logger.info('Save log to {}'.format(logger_path))
    # print configuration
    logger.info("--------------- Configuration ---------------")
    logger.info("\n%s", yaml.dump(config, sort_keys=False, indent=4))

    if config["wandb"] and thread_flag(config["ddp"]):
        os.makedirs(config["wandb_dir"], exist_ok=True)
        run = wandb.init(project=config["wandb_proj"], config=config, tags=[args.wandb_tags],
                             dir=config["wandb_dir"])
        run.name = args.wandb_tags

    # prepare the data
    val_data_loader = prepare_val_data(config)
    test_data_loaders = prepare_testing_data(config)
    train_data_loader, num_ids = prepare_training_data(config)
    config["num_ids"] = num_ids

    # prepare the model (detector)
    print("before_model")
    model_class = DETECTOR[config['model_name']]
    print(model_class)
    model = model_class(config)
    print("after_model")
    if thread_flag(config["ddp"]):
        check_trainable_params(model)
        getModelSize(model)

    # prepare the optimizer
    if config["optimizer"].get("split", False) and config["optimizer"]["split"]:
        optimizer, optimizer_prompt = choose_optimizer_split_prompt(model, config)
        scheduler, scheduler_prompt = choose_scheduler(config, optimizer), choose_scheduler(config, optimizer_prompt)
    else:
        optimizer = choose_optimizer(model, config)
        scheduler = choose_scheduler(config, optimizer)
        optimizer_prompt, scheduler_prompt = None, None
    swa_model = optim.swa_utils.AveragedModel(model) if config["SWA"] else None
    
    # prepare the metric
    metric_scoring = choose_metric(config)

    # prepare the trainer
    trainer = Trainer(config, model, optimizer, scheduler, logger, metric_scoring, logger_path, swa_model, 
                      optimizer_prompt=optimizer_prompt, scheduler_prompt=scheduler_prompt)

    # start training
    for epoch in range(config['start_epoch'], config['nEpochs'] + 1):
        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
            test_data_loaders=test_data_loaders
        )
    logger.info("Training done, best val metric {}".format(parse_metric_for_print(best_metric))) 


if __name__ == '__main__':
    main()
