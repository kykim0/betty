import argparse
import copy
import logging
import math
import os

from betty.configs import Config, EngineConfig
from betty.engine import Engine
from betty.problems import ImplicitProblem
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import transformers
from transformers import get_linear_schedule_with_warmup

from bonito import get_accelerate_model, get_weight_model
from configs import DataArguments, ModelArguments, TrainingArguments
from data import get_dataloader, make_data_module
from utils import set_seed


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class WeightedLoss(object):

    def __call__(self, model_output, labels, weights=None, shift_labels=False, average_loss=True):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        if shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='none')
        loss = loss.view(logits.size(0), logits.size(1)).mean(axis=1)
        if weights is not None:
            if weights.dim() > 1:
                weights = weights.squeeze(-1)
            loss = loss * weights
        return loss.mean() if average_loss else loss


class Outer(ImplicitProblem):

    def __init__(self, training_args, name, config,
                 module=None, optimizer=None, scheduler=None,
                 train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer, scheduler,
                         train_data_loader, extra_config)
        self.args = training_args

    def forward(self, inputs):
        # print(f"[outer] inputs device: {inputs['input_ids'].device} | module device: {self.module.model.device}")
        return self.module(**inputs)

    def training_step(self, batch):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        labels = batch["labels"]
        outputs = self.inner(inputs)
        # outputs = labels

        loss_fn = WeightedLoss()
        loss = loss_fn(outputs, labels, weights=None, shift_labels=True)
        return loss

    def configure_optimizer(self):
        self.meta_optimizer = optim.Adam(
            self.module.parameters(),
            lr=self.args.weight_net_lr,
            weight_decay=self.args.weight_net_decay,
        )
        return self.meta_optimizer


# TODO(kykim): The "Expected all tensors to be on the same device" issue seems
# to have to do with this.
class Inner(ImplicitProblem):

    def __init__(self, training_args, name, config,
                 module=None, optimizer=None, scheduler=None,
                 train_data_loader=None, extra_config=None):
        super().__init__(name, config, module, optimizer, scheduler,
                         train_data_loader, extra_config)
        self.args = training_args

    def forward(self, inputs):
        # print(f"[inner] inputs device: {inputs['input_ids'].device} | module device: {self.module.device}")
        return self.module(**inputs)

    def training_step(self, batch):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        labels = batch["labels"]
        outputs = self.forward(inputs)

        wnet_inputs = {
            "input_ids": batch["wnet_input_ids"],
            "attention_mask": batch["wnet_attention_mask"],
        }
        weights = self.outer(wnet_inputs)

        loss_fn = WeightedLoss()
        # loss = loss_fn(outputs, labels, weights=None, shift_labels=True)
        loss = loss_fn(outputs, labels, weights=weights, shift_labels=True)
        return loss

    def configure_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay
        )
        return self.optimizer

    def configure_scheduler(self):
        train_steps = self.args.max_steps
        warmup_steps = (
            self.args.warmup_steps if self.args.warmup_steps > 0 else math.ceil(train_steps * self.args.warmup_ratio)
        )
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup_steps, train_steps)
        return self.scheduler


class ReweightingEngine(Engine):

    def __init__(self, training_args, test_dataloader, problems,
                 config=None, dependencies=None, env=None):
        super().__init__(problems, config, dependencies, env)
        self.args = training_args
        self.test_dataloader = test_dataloader
        self._best_val_loss = np.inf

    @torch.no_grad()
    def validation(self):
        total_loss = 0.0
        num_examples = 0
        for batch in self.test_dataloader:
            num_examples += len(batch["input_ids"])
            inputs = {
                "input_ids": batch["wnet_input_ids"].to(self.device),
                "attention_mask": batch["wnet_attention_mask"].to(self.device),
            }
            labels = batch["labels"].to(self.device)
            with torch.no_grad():
                outputs = self.inner(inputs)
            loss_fn = WeightedLoss()
            loss = loss_fn(outputs, labels, weights=None, shift_labels=True, average_loss=False)
            total_loss += loss.sum()

        val_loss = total_loss / num_examples
        if self._best_val_loss > val_loss:
            self._best_val_loss = val_loss
            if self.is_rank_zero():
                output_dir = self.args.output_dir
                os.makedirs(output_dir, exist_ok=True)
                torch.save(self.inner.state_dict(), f"{output_dir}/model_{self.global_step}.pt")
                torch.save(self.outer.state_dict(), f"{output_dir}/wnet_{self.global_step}.pt")
        return {"loss": val_loss, "best_acc": self._best_val_loss}


def main():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        _,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    model, tokenizer = get_accelerate_model(args, args.checkpoint_model_id_or_path)
    # model.config.use_cache = False if args.gradient_checkpointing else True  # Needed?
    wnet_model, wnet_tokenizer = get_weight_model(args)
    print("loaded model")
    set_seed(args.seed)

    data_module = make_data_module(tokenizer, wnet_tokenizer, args)

    num_gpus = torch.cuda.device_count()
    # train_batch_size = num_gpus * args.per_device_train_batch_size * args.gradient_accumulation_steps
    train_batch_size = 1
    train_dataloader = get_dataloader(
        dataset=data_module["train_dataset"],
        data_collator=data_module["collator"],
        batch_size=train_batch_size,
        group_by_length=args.group_by_length,
    )
    # meta_batch_size = num_gpus * args.per_device_meta_train_batch_size * args.meta_gradient_accumulation_steps
    eval_dataset = data_module["eval_dataset"]
    dataset = eval_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
    meta_batch_size = 1
    meta_dataloader = get_dataloader(
        dataset=dataset["train"],
        data_collator=data_module["collator"],
        batch_size=meta_batch_size,
        group_by_length=args.group_by_length,
    )
    test_batch_size = 32
    test_dataloader = get_dataloader(
        dataset=dataset["test"],
        data_collator=data_module["collator"],
        batch_size=test_batch_size,
        group_by_length=args.group_by_length,
    )

    outer_config = Config(type="darts", precision="bf16", log_step=args.logging_steps,
                          retain_graph=True, gradient_accumulation=args.gradient_accumulation_steps)
    inner_config = Config(type="darts", precision="bf16", unroll_steps=1,
                          gradient_accumulation=args.meta_gradient_accumulation_steps)
    args.report_to = args.report_to if args.report_to in ["tensorboard", "wandb", "none"] else "none"
    engine_config = EngineConfig(
        train_iters=args.max_steps,
        valid_step=args.eval_steps,
        strategy=args.strategy,
        roll_back=False,
        logger_type=args.report_to,
    )
    outer = Outer(args, name="outer", config=outer_config, module=wnet_model, train_data_loader=meta_dataloader)
    inner = Inner(args, name="inner", config=inner_config, module=model, train_data_loader=train_dataloader)

    # if args.baseline or args.retrain:
    #     problems = [inner]
    #     u2l, l2u = {}, {}
    problems = [outer, inner]
    u2l = {outer: [inner]}
    l2u = {inner: [outer]}
    dependencies = {"l2u": l2u, "u2l": u2l}

    engine = ReweightingEngine(
        args, test_dataloader, problems=problems, config=engine_config, dependencies=dependencies
    )
    engine.run()


if __name__ == "__main__":
    # TODO(kykim) Find a better way.
    # if not training_args.max_steps > 0:
    #     training_args.num_train_epochs = 1
    # training_args.save_strategy = "steps"
    # training_args.save_steps = 1000
    # training_args.save_only_model = True
    # training_args.gradient_accumulation_steps = 1
    # training_args.per_device_train_batch_size = 2
    # training_args.per_device_meta_train_batch_size = 2
    main()
