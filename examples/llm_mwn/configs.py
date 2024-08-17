
from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        # default="mistralai/Mistral-7B-v0.1",
        default="microsoft/Phi-3-mini-4k-instruct",
        # default="EleutherAI/pythia-2.8b",
    )
    trust_remote_code: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_max_len: int = field(
        default=1024,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    training_type: str = field(
        default="instruct",
        metadata={"help": "Either instruct or bonito_training"},
    )
    supervision_source: str = field(
        default="bonito",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )
    dataset: str = field(
        default="pubmed_qa",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )
    custom_dataset_path: str = field(
        default=None,
        metadata={"help": "Path to a custom dataset to use for training."},
    )

    def __post_init__(self):
        valid_training_types = ["instruct", "bonito_training"]
        valid_supervision_sources = [
            "bonito",
            "dapt",
            "mistral_instruct",
            "zephyr_beta",
            "p3",
        ]

        if self.training_type not in valid_training_types:
            raise ValueError(
                f"training_type must be one of {valid_training_types}, got '{self.training_type}'"
            )

        if self.supervision_source not in valid_supervision_sources:
            raise ValueError(
                f"supervision_source must be one of {valid_supervision_sources}, got '{self.supervision_source}'"
            )


@dataclass
class TrainingArguments:
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    report_to: str = field(
        default="wandb",
        # default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    checkpoint_model_id_or_path: str = field(
        default=None, metadata={"help": "the pretrained checkpoint dir to load from"}
    )
    huggingface_checkpoint: str = field(
        default=None, metadata={"help": "the pretrained checkpoint dir to load from"}
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    max_steps: Optional[int] = field(
        default=10000, metadata={"help": "How many optimizer update steps to take"}
    )
    eval_steps: Optional[int] = field(
        default=1000, metadata={"help": "How many optimizer update steps to take before an eval."}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "How many epochs to train for"}
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0001, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    ddp_find_unused_parameter: bool = field(
        default=False, metadata={"help": "Find unused parameters in DDP training."}
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=100,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(default="steps", metadata={"help": "When to save checkpoints"})
    save_steps: Optional[int] = field(
        default=10000, metadata={"help": "How often to save a model"}
    )
    save_total_limit: int = field(
        default=10,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    strategy: str = field(default="default", metadata={"help": "One of 'default', 'zero', 'fsdp'"})
    precision: str = field(default="bf16", metadata={"help": "Precision to use."})
    seed: int = field(default=42, metadata={"help": "Seed."})

    # Args realted to meta-weight-net training.
    meta_step_interval: int = field(
        default=1,
        metadata={
            "help": "The number of inner optimization steps before running an outer optimization."
        },
    )
    per_device_meta_train_batch_size: int = field(
        default=1, metadata={"help": "Per device batch size for meta training dataset."}
    )
    meta_gradient_accumulation_steps: int = field(
        default=4,
        metadata={
            "help": "How many gradients to accumulate before to perform a meta optimizer step"
        },
    )
    weight_net_lr: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for weight-net."},
    )
    weight_net_scheduler: str = field(
        default="linear",
        metadata={"help": "Learning rate scheduler for weight-net."},
    )
    weight_net_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for weight-net if we apply some."},
    )
    weight_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Name of the weight model to use, e.g., EleutherAI/pythia-160m."},
    )
