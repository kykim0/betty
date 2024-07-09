import argparse
import os
import math
import torch
import numpy as np

from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from betty.engine import Engine

from data_loader import *
from model import *
from util import set_seed, acc_and_f1

from transformers import (
    SchedulerType,
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
)

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


parser = argparse.ArgumentParser(description="TAPT")
# parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="The name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default=None,
    help="The configuration name of the dataset to use (via the datasets library).",
)
parser.add_argument(
    "--map_labels_to_ids",
    action="store_true",
    help="If passed, will map string labels to numeric ids.",
)
parser.add_argument(
    "--train_file",
    type=str,
    default=None,
    help="A csv or a json file containing the training data.",
)
parser.add_argument(
    "--validation_file",
    type=str,
    default=None,
    help="A csv or a json file containing the validation data.",
)
parser.add_argument(
    "--pad_to_max_length",
    action="store_true",
    help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
)
parser.add_argument(
    "--model_name_or_path",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=True,
)
parser.add_argument(
    "--config_name",
    type=str,
    default=None,
    help="Pretrained config name or path if not the same as model_name",
)
parser.add_argument(
    "--tokenizer_name",
    type=str,
    default=None,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--use_slow_tokenizer",
    action="store_true",
    help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
)
parser.add_argument(
    "--per_device_train_batch_size",
    type=int,
    default=16,
    help="Batch size (per device) for the training dataloader.",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    type=int,
    default=32,
    help="Batch size (per device) for the evaluation dataloader.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=2e-5,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--classifier_dropout",
    type=float,
    default=0.1,
    help="Dropout applied for the feedforward layer of the classifier.",
)
parser.add_argument(
    "--adam_beta1",
    type=float,
    default=0.9,
    help="Beta1 for Adam optimizer.",
)
parser.add_argument(
    "--adam_beta2",
    type=float,
    default=0.98,
    help="Beta2 for Adam optimizer.",
)
parser.add_argument(
    "--adam_epsilon",
    type=float,
    default=1e-6,
    help="Epsilon for Adam optimizer.",
)
parser.add_argument(
    "--weight_decay", type=float, default=0.0, help="Weight decay to use."
)
parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=3,
    help="Total number of training epochs to perform.",
)
parser.add_argument(
    "--max_train_steps",
    type=int,
    default=0,
    help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--lr_scheduler_type",
    type=SchedulerType,
    default="linear",
    help="The scheduler type to use.",
    choices=[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ],
)
parser.add_argument(
    "--num_warmup_steps",
    type=int,
    default=0,
    help="Number of steps for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--warmup_proportion",
    type=float,
    default=0.06,
    help="Fraction of total steps for the warmup in the lr scheduler.",
)
parser.add_argument(
    "--lmbda",
    type=float,
    default=1.0,
    help="Weight for the mlm loss.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default=None,
    help="Where to store the final model.",
    required=True,
)
parser.add_argument(
    "--seed", type=int, default=1, help="A seed for reproducible training."
)
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help="Model type to use if training from scratch.",
    choices=MODEL_TYPES,
)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=512,
    help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
)
parser.add_argument(
    "--line_by_line",
    type=bool,
    default=False,
    help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
)
parser.add_argument(
    "--preprocessing_num_workers",
    type=int,
    default=None,
    help="The number of processes to use for the preprocessing.",
)
parser.add_argument(
    "--overwrite_cache",
    type=bool,
    default=False,
    help="Overwrite the cached training and evaluation sets",
)
parser.add_argument(
    "--mlm_probability",
    type=float,
    default=0.15,
    help="Ratio of tokens to mask for masked language modeling loss",
)
parser.add_argument(
    "--best_metric",
    type=str,
    default="acc",
    help="Model selection metric.",
    choices=["acc", "f1"],
)
parser.add_argument(
    "--f1_average",
    type=str,
    default="macro",
    help="Averaging method for f1.",
    choices=["macro", "micro"],
)
parser.add_argument(
    "--label_field_name",
    type=str,
    default="label",
    help="Label field name in HF datasets.",
    choices=["label", "intent"],
)


parser.add_argument("--mode", type=str, default="baseline")
parser.add_argument("--reweight_strategy", type=str, default="loss")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
# parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--meta_lr", type=float, default=1e-5)
parser.add_argument("--gradient_clipping", type=float, default=1.0)
# parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--unroll_steps", type=int, default=5)

parser.add_argument("--strategy", type=str, default="default")
parser.add_argument("--train_iters", type=int, default=750)
parser.add_argument("--valid_step", type=int, default=50)
parser.add_argument(
    "--roll_back", type=bool, default=False, help="Whether to roll back!"
)
parser.add_argument(
    "--global_weight",
    type=bool,
    default=False,
    help="Whether to meta-learn global weight!",
)
parser.add_argument(
    "--zeroinit_output_layer",
    type=bool,
    default=False,
    help="Whether to zero-init output layer weights and bias!",
)
parser.add_argument("--darts_adam_alpha", type=float, default=1.0)

# parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

print(args)

set_seed(args.seed)

if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)

if args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name, use_fast=not args.use_slow_tokenizer
    )
elif args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=not args.use_slow_tokenizer
    )
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )

mlm_dataloaders = get_mlm_dataloaders(args=args, tokenizer=tokenizer)
task_dataloaders, label_list, num_labels = get_task_dataloaders(
    args=args, tokenizer=tokenizer
)

if args.model_name_or_path:
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        classifier_dropout=args.classifier_dropout,
    )
else:
    config = CONFIG_MAPPING[args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")

base_model = BaseModel.from_pretrained(
    args.model_name_or_path,
    from_tf=bool(".ckpt" in args.model_name_or_path),
    config=config,
)

base_model.resize_token_embeddings(len(tokenizer))

if label_list is not None:
    base_model.config.label2id = {l: i for i, l in enumerate(label_list)}
    base_model.config.id2label = {id: label for label, id in config.label2id.items()}


base_cls_loader = task_dataloaders["train"]
val_base_cls_loader = task_dataloaders["validation"]
test_base_cls_loader = task_dataloaders["test"]
base_mlm_loader = mlm_dataloaders["train"]
# base_model = None
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [
            p
            for n, p in base_model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": args.weight_decay,
    },
    {
        "params": [
            p
            for n, p in base_model.named_parameters()
            if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

num_update_steps_per_epoch = (
    len(task_dataloaders["train"]) // args.gradient_accumulation_steps
)
num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

if args.max_train_steps > 0:
    max_train_steps = args.max_train_steps
    num_train_epochs = args.max_train_steps // num_update_steps_per_epoch + int(
        args.max_train_steps % num_update_steps_per_epoch > 0
    )
else:
    max_train_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    num_train_epochs = math.ceil(args.num_train_epochs)

print("Gradient accumulation steps : ", args.gradient_accumulation_steps)
print("Number of update steps per epoch : ", num_update_steps_per_epoch)
print("Max number of train steps : ", max_train_steps)

# We evaluate after every epoch
valid_step = num_update_steps_per_epoch
# valid_step=10
train_iters = max_train_steps

base_optimizer = AdamW(
    base_model.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    eps=args.adam_epsilon,
)
num_warmup_steps = int(args.warmup_proportion * max_train_steps)
print("Number of warmup steps : ", num_warmup_steps)
base_scheduler = get_linear_schedule_with_warmup(
    base_optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)
base_config = Config(
    type="darts_adam",
    retain_graph=True,
    gradient_accumulation=args.gradient_accumulation_steps,
    gradient_clipping=args.gradient_clipping,
    unroll_steps=args.unroll_steps,
    darts_adam_alpha=args.darts_adam_alpha,
)


class Base(ImplicitProblem):
    def training_step(self, batch):

        cls_batch, mlm_batch = batch
        # import ipdb; ipdb.set_trace()
        cls_output = self.module(**cls_batch, use_classifier_head=True)["classifier"]
        # mlm_output = self.module(**mlm_batch, use_mlm_head=True)['mlm']
        # cls_loss, mlm_loss, mlm_embedding = self.module(cls_batch, mlm_batch)
        if args.mode == "baseline":
            total_loss = cls_output.loss

        elif args.mode == "multitask":
            mlm_output = self.module(**mlm_batch, use_mlm_head=True)["mlm"]
            mlm_loss_final = args.lmbda * mlm_output.loss
            total_loss = cls_output.loss + mlm_loss_final

        elif args.mode == "meta":
            mlm_output = self.module(**mlm_batch, use_mlm_head=True, reduction="none")[
                "mlm"
            ]

            if args.reweight_strategy == "loss":
                weight = self.meta(mlm_output.loss.detach().view(-1, 1))
            else:
                weight = self.meta(mlm_output.hidden_states.detach())
            mlm_loss_final = torch.mean(weight * mlm_output.loss)
            total_loss = cls_output.loss + mlm_loss_final

        else:
            raise ValueError("Invalid mode")

        return total_loss


base = Base(
    name="base",
    module=base_model,
    optimizer=base_optimizer,
    scheduler=base_scheduler,
    train_data_loader=(base_cls_loader, base_mlm_loader),
    config=base_config,
)


class Meta(ImplicitProblem):
    def training_step(self, batch):
        loss = self.base(**batch, use_classifier_head=True)["classifier"].loss

        return loss


meta_dataloaders, label_list, num_labels = get_meta_dataloaders(
    args=args, tokenizer=tokenizer
)
meta_loader = meta_dataloaders["train"]
if args.reweight_strategy == "loss":
    meta_model = MLP(
        in_size=1,
        global_weight=args.global_weight,
        zeroinit_output_layer=args.zeroinit_output_layer,
    )
else:
    meta_model = MLP(
        in_size=768,
        global_weight=args.global_weight,
        zeroinit_output_layer=args.zeroinit_output_layer,
    )

meta_optimizer = torch.optim.Adam(
    meta_model.parameters(),
    lr=args.meta_lr,
    betas=(args.adam_beta1, args.adam_beta2),
    eps=args.adam_epsilon,
)
meta_config = Config(retain_graph=True)
meta = Meta(
    name="meta",
    module=meta_model,
    optimizer=meta_optimizer,
    train_data_loader=meta_loader,
    config=meta_config,
)


class TAPTEngine(Engine):
    @torch.no_grad()
    def validation(self):

        if not hasattr(self, "best_acc"):
            self.best_acc = -1
        if not hasattr(self, "best_f1"):
            self.best_f1 = -1
        if not hasattr(self, "best_metric"):
            self.best_metric = -1
        if not hasattr(self, "best_test_acc"):
            self.best_test_acc = -1
        if not hasattr(self, "best_test_f1"):
            self.best_test_f1 = -1
        if not hasattr(self, "best_test_metric"):
            self.best_test_metric = -1

        preds = None
        labels = None

        for step, batch in enumerate(val_base_cls_loader):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.base.device)
            outputs = self.base(**batch, use_classifier_head=True)["classifier"]
            predictions = outputs.logits.argmax(dim=-1)

            if preds is None:
                preds = predictions.cpu().numpy()
                labels = batch["labels"].cpu().numpy()
            else:
                preds = np.concatenate((preds, predictions.cpu().numpy()))
                labels = np.concatenate((labels, batch["labels"].cpu().numpy()))

        metrics = acc_and_f1(preds=preds, labels=labels, average=args.f1_average)

        test_preds = None
        test_labels = None

        for step, batch in enumerate(test_base_cls_loader):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.base.device)
            outputs = self.base(**batch, use_classifier_head=True)["classifier"]
            predictions = outputs.logits.argmax(dim=-1)

            if test_preds is None:
                test_preds = predictions.cpu().numpy()
                test_labels = batch["labels"].cpu().numpy()
            else:
                test_preds = np.concatenate((test_preds, predictions.cpu().numpy()))
                test_labels = np.concatenate(
                    (test_labels, batch["labels"].cpu().numpy())
                )

        test_metrics = acc_and_f1(
            preds=test_preds, labels=test_labels, average=args.f1_average
        )

        if metrics["acc"] > self.best_acc:
            self.best_acc = metrics["acc"]
            self.best_test_acc = test_metrics["acc"]

        if metrics["f1"] > self.best_f1:
            self.best_f1 = metrics["f1"]
            self.best_test_f1 = test_metrics["f1"]

        if args.best_metric == "acc":
            self.best_metric = self.best_acc
            self.best_test_metric = self.best_test_acc
        else:
            self.best_metric = self.best_f1
            self.best_test_metric = self.best_test_f1

        return {
            "acc": metrics["acc"],
            "best_acc": self.best_acc,
            "f1": metrics["f1"],
            "best_f1": self.best_f1,
            "best_metric": self.best_metric,
            "test_acc": test_metrics["acc"],
            "best_test_acc": self.best_test_acc,
            "test_f1": test_metrics["f1"],
            "best_test_f1": self.best_test_f1,
            "best_test_metric": self.best_test_metric,
        }


engine_config = EngineConfig(
    train_iters=train_iters,
    valid_step=valid_step,
    strategy=args.strategy,
    roll_back=args.roll_back,
)


if args.mode != "meta":
    problems = [base]
    u2l, l2u = {}, {}
else:
    problems = [base, meta]
    u2l, l2u = {meta: [base]}, {base: [meta]}
dependencies = {"l2u": l2u, "u2l": u2l}

engine = TAPTEngine(
    config=engine_config,
    problems=problems,
    dependencies=dependencies,
)
engine.run()
