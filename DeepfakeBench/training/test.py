"""
eval pretained model.
"""
import os
import argparse
import datetime
import yaml
from copy import deepcopy
import torch
import torch.nn.parallel
import torch.utils.data

from trainer.trainer import Trainer
from detectors import DETECTOR

from dataset import prepare_testing_data
from utils.metrics import choose_metric
from utils.logger import create_logger
from utils.config import get_config
from utils.model_utils import set_seed, thread_flag
import wandb
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path



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
parser.add_argument('--weights_path', type=str, default='')
parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    # config
    config = get_config(args)
    weights_path, logger_path = None, None
    if args.weights_path:
        config['weights_path'] = args.weights_path
        weights_path = args.weights_path
        if config["model_name"] in ["aide", "npr", "freqnet"]:
            logger_path = Path(weights_path).parents[0]
        else:
            logger_path = Path(weights_path).parents[2]

    # create logger
    if not logger_path:
        timenow=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        logger_path = os.path.join(
            config['log_dir'],
            config['train_dataset'][0],
            timenow
        )
        os.makedirs(logger_path, exist_ok=True)
    logger = create_logger(os.path.join(logger_path, 'testing.log'))
    logger.info('Save log to {}'.format(logger_path))
    # print configuration
    logger.info("--------------- Configuration ---------------")
    logger.info("\n%s", yaml.dump(config, sort_keys=False, indent=4))

    if config["wandb"] and thread_flag(config["ddp"]):
        os.makedirs(config["wandb_dir"], exist_ok=True)
        run = wandb.init(project=config["wandb_proj"], config=config, tags=[args.wandb_tags],
                             dir=config["wandb_dir"])
        run.name = args.wandb_tags
        
    # init seed
    set_seed(config)

    # prepare the model (detector)
    config["num_ids"] = 13943
    model_class = DETECTOR[config['model_name']]
    model = model_class(config)
    org_cfg = deepcopy(config)
    try:
        config["test_transform_cospy"] = model.backbone.model.test_transform
    except:
        config = org_cfg

    # prepare the testing data loader
    test_data_loaders = prepare_testing_data(config)
    metric_scoring = choose_metric(config)
    
    # load the pre-trained weights
    if weights_path:
        ckpt = torch.load(weights_path, map_location=device)
        try:
            model.load_state_dict(ckpt, strict=True)
        except:
            # other small models, use specific loading func
            model.load_ckpt(ckpt)
        # model.load_state_dict(ckpt, strict=True)
        logger.info('===> Load checkpoint from: {}'.format(weights_path))
    else:
        logger.info('Fail to load the trained weights')

    trainer = Trainer(config, model, logger=logger, metric_scoring=metric_scoring, log_dir=logger_path)
    
    logger.info("Testing")
    test_best_metric = trainer.test_epoch(
        test_data_loaders,
        phase='val'
    )


if __name__ == '__main__':
    main()
