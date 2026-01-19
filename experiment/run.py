import csv
import json
import os
import time
from collections import OrderedDict
from datetime import datetime
from statistics import mean

import numpy as np
import torch
import yaml
from timm.utils import get_outdir

from engine.trainer import Trainer
from experiment.build_loader import get_loader
from experiment.build_model import get_model
from utils.global_var import OUTPUT_DIR, TUNE_DIR, TUNE_DIR_TEST
from utils.log_utils import logging_env_setup
from utils.misc import method_name, set_seed
from utils.setup_logging import get_logger

logger = get_logger("Prompt_CAM")


def train(params, train_loader, val_loader, test_loader):
    model, tune_parameters, model_grad_params_no_head = get_model(params)
    trainer = Trainer(model, tune_parameters, params)
    train_metrics, best_eval_metrics, eval_metrics = trainer.train_classifier(
        train_loader, val_loader, test_loader
    )
    return (
        train_metrics,
        best_eval_metrics,
        eval_metrics,
        model_grad_params_no_head,
        trainer.model,
    )


def basic_run(params):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data_name = params.data.split("-")[-1]
    dataset_name = params.data.split("-")[0]
    method = method_name(params)
    start_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if hasattr(params, "output_dir") and params.output_dir:
        output_dir = params.output_dir
    else:
        output_dir = os.path.join(
            OUTPUT_DIR,
            params.pretrained_weights,
            dataset_name,
            method,
            data_name,
            start_time,
        )
    params.output_dir = get_outdir(output_dir)
    params_text = yaml.safe_dump(params.__dict__, default_flow_style=False)
    with open(os.path.join(params.output_dir, "args.yaml"), "w") as f:
        f.write(params_text)
    logging_env_setup(params)
    logger.info(f"Start loading {data_name}")
    train_loader, val_loader, test_loader = get_loader(params, logger)

    train(params, train_loader, val_loader, test_loader)


def update_output_dir(default_params, test):
    logger.info(f"start running {default_params.method_name}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    data_name = default_params.data.split("-")[-1]
    dataset_name = default_params.data.split("-")[0]
    method = default_params.method_name
    if test:
        output_dir = os.path.join(
            TUNE_DIR_TEST,
            default_params.experiment_name,
            dataset_name,
            data_name,
            method,
        )
    else:
        output_dir = os.path.join(
            TUNE_DIR, default_params.experiment_name, dataset_name, data_name, method
        )
    default_params.output_dir = output_dir

    logging_env_setup(default_params)
    return output_dir, data_name


def evaluate(default_params):
    _, _, test_loader = get_loader(default_params, logger)
    if "eval" in default_params.test_data:
        result_name = f"{default_params.test_data.split('_')[1]}_result.json"
    else:
        result_name = f"{default_params.test_data}_result.json"
    if not os.path.isfile(os.path.join(default_params.output_dir, result_name)):
        if not os.path.isfile(
            os.path.join(default_params.output_dir, "final_result.json")
        ):
            logger.info(
                "no final_result.json, the model is not fine-tuned, show model zero shot performance"
            )
            best_tune = ()
            result_name = "zero_shot_" + result_name
        else:
            result = json.load(
                open(os.path.join(default_params.output_dir, "final_result.json"))
            )
            best_tune = result["best_tune"]
            default_params.update(best_tune)

        model, tune_parameters, model_grad_params_no_head = get_model(default_params)
        trainer = Trainer(model, tune_parameters, default_params)
        if not os.path.isfile(os.path.join(default_params.output_dir, "model.pt")):
            assert not os.path.isfile(
                os.path.join(default_params.output_dir, "final_result.json")
            )
            logger.info("no model.pt, shows zero shot performance")
        else:
            trainer.load_weight()
        eval_metrics = trainer.eval_classifier(test_loader, "test")
        json.dump(
            {
                "avg_acc": eval_metrics["top1"],
                "inserted_parameters": model_grad_params_no_head,
                "best_tune": best_tune,
            },
            open(os.path.join(default_params.output_dir, result_name), "w"),
        )
    else:
        logger.info(f"finish {result_name} for {default_params.method_name}")
    return


def result_tracker(
    first_col,
    train_metrics,
    eval_metrics,
    best_eval_metrics,
    filename,
    write_header=False,
    first_col_name="param_set",
    eval_name="val_",
):
    rowd = OrderedDict([(first_col_name, first_col)])
    rowd.update([("train_" + k, v) for k, v in train_metrics.items()])
    rowd.update([(eval_name + k, v) for k, v in eval_metrics.items()])
    rowd.update([(eval_name + "best_" + k, v) for k, v in best_eval_metrics.items()])
    with open(filename, mode="a") as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:
            dw.writeheader()
        dw.writerow(rowd)
