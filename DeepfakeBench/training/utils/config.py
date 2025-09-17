import yaml
import argparse


def parse_value(value, original_value=None):
    if isinstance(original_value, bool):
        return value.lower() in {"true", "1"}
    elif isinstance(original_value, int):
        return int(value)
    elif isinstance(original_value, float):
        return float(value)
    elif isinstance(original_value, list):
        return value.split(",")
    return value


def update_cfg_from_opts(cfg_dict, opts):
    for i in range(0, len(opts), 2):
        keys = opts[i].split(".")  # 键以点号分隔
        new_value = opts[i + 1]
        d = cfg_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})  # 确保嵌套字典存在
        # 获取原始值（如果存在），解析新值
        if keys[-1] not in d:
            raise ValueError(f"Unknown config parameters: {opts[i]}")
        else:
            original_value = d.get(keys[-1], None)
            d[keys[-1]] = parse_value(new_value, original_value)


def get_config(args):
    # parse options and load config
    with open(args.detector_cfg, 'r') as f:
        config = yaml.safe_load(f)
    with open(args.dataset_cfg, 'r') as f:
        config2 = yaml.safe_load(f)

    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['local_rank'] = args.local_rank
    try:
        config['world_size'] = args.world_size
    except:
        config['world_size'] = 1
    if config['dry_run']:
        config['nEpochs'] = 0
        config['save_feat']=False
    # If arguments are provided, they will overwrite the yaml settings
    if args.train_dataset:
        config['train_dataset'] = args.train_dataset
    if args.val_dataset:
        config['val_dataset'] = args.val_dataset
    if args.test_dataset:
        config['test_dataset'] = args.test_dataset
    if config['lmdb']:
        config['dataset_json_folder'] = 'preprocessing/dataset_json_v3'
    config['ddp']= args.ddp

    # update from arg opts
    if args.opts:
        update_cfg_from_opts(config, args.opts)

    return config
