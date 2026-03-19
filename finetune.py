import os
import json
from typing import Dict
from functools import partial

from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter

from transformers import Trainer,TrainingArguments
from transformers.utils import logging,is_sagemaker_mp_enabled

import argparse
import dataclasses

from utils.args import parse_train_args,parse_global_args,parse_dataset_args,parse_model_args
from utils.loader import load_models_tokenizer, load_datasets

from collator import Collator

logging.set_verbosity(logging.INFO)
logger = logging.get_logger(__name__)

str2list = lambda s: list(map(int, s.split(",")))

class Custom_Trainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        SCALE1 = 1
        SCALE2 = 100
        SCALE3 = 10
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ('mtp_head' not in n and n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ('logit_head' in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": SCALE1*self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ('token_emb' in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay/1000,
                    "lr": SCALE2*self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ('rnn' in n or 'transition' in n and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay/1000,
                    "lr": SCALE3*self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if ('mtp_head' not in n and n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
            ]
            print('optimizer_grouped_parameters Inited')
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)

            # Overwrite `params` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for GaLore optimizer.
            if "params" in optimizer_kwargs:
                print("Hack PARAMS 1")
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")

            # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
            # e.g. for LOMO optimizer.
            if "model" in optimizer_kwargs:
                print("Hack PARAMS 2")
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")

            # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
            # to avoid arguments conflicts.
            if "optimizer_dict" in optimizer_kwargs:
                print("Hack PARAMS 3")
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        if is_sagemaker_mp_enabled():
            print("sagemaker_enabled")
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


class Tuner(object):
    def __init__(self,args):
        self.args = args

    def prepare_model(self):
        model, tokenizer = load_models_tokenizer(
            self.args, is_eval=False
        )
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        print("Model prepared: \n{}".format(model))
        self.model = model
        self.tokenizer = tokenizer

    def train(self):
        self.args.remove_unused_columns = False
        collator = Collator(args, self.tokenizer)
        train_data, valid_data = load_datasets(args)
        args_dict = vars(args)
        training_arg_names = {f.name for f in dataclasses.fields(TrainingArguments)}
        training_args_dict = {
            key: value for key, value in args_dict.items() if key in training_arg_names
        }
        training_args = TrainingArguments(**training_args_dict)
        if not self.args.use_mtp:
            trainer = Trainer(
                model=self.model,
                train_dataset=train_data,
                eval_dataset=valid_data,
                data_collator=collator,
                processing_class=self.tokenizer,
                args=training_args,
            )
        else:
            trainer = Custom_Trainer(
                model=self.model,
                train_dataset=train_data,
                eval_dataset=valid_data,
                data_collator=collator,
                processing_class=self.tokenizer,
                args=training_args,
            )
        trainer.train(resume_from_checkpoint=self.args.resume_from_checkpoint)
        trainer.save_state()
        trainer.save_model(output_dir=args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)
    parser = parse_model_args(parser)
    args = parser.parse_args()
    args.docid_num=str2list(args.docid_num)
    args.num_train_epochs=args.epochs
    args.per_device_train_batch_size=args.per_device_batch_size
    args.per_device_eval_batch_size=args.per_device_batch_size
    args.eval_strategy=args.save_and_eval_strategy
    args.save_strategy=args.save_and_eval_strategy
    args.logging_steps=args.logging_step
    args.eval_steps=args.save_and_eval_steps
    args.save_steps=args.save_and_eval_steps
    args.save_total_limit=2
    args.load_best_model_at_end=True
    tuner = Tuner(args)
    print('----args----')
    print(tuner.args)

    tuner.prepare_model()
    tuner.train()