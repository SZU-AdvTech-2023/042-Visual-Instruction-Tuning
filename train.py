import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import wandb

import VQABBoxpro.tasks as tasks
from VQABBoxpro.common.config import Config
from VQABBoxpro.common.dist_utils import get_rank, init_distributed_mode
from VQABBoxpro.common.logger import setup_logger
from VQABBoxpro.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from VQABBoxpro.common.registry import registry
from VQABBoxpro.common.utils import now

# imports modules for registration
from VQABBoxpro.datasets.builders import *
from VQABBoxpro.models import *
from VQABBoxpro.processors import *
from VQABBoxpro.runners import *
from VQABBoxpro.tasks import *


def get_args():
    arg_parser = argparse.ArgumentParser(description="Training")

    arg_parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    arg_parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parsed_args = arg_parser.parse_args()

    return parsed_args


def set_seeds(config):
    seed_value = config.run_cfg.seed + get_rank()

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    cudnn.benchmark = False
    cudnn.deterministic = True


def fetch_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_class = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_class


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_identifier = now()
    command_args = get_args()
    cfg = Config(command_args)

    init_distributed_mode(cfg.run_cfg)
    set_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    if cfg.run_cfg.wandb_log:
        wandb.login()
        wandb.init(project="minigptv", name=cfg.run_cfg.job_name)
        wandb.watch(model)

    runner = fetch_runner_class(cfg)(
        cfg=cfg, job_id=job_identifier, task=task, model=model, datasets=datasets
    )
    runner.train()


if __name__ == "__main__":
    main()