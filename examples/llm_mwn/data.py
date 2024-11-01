
import copy
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import datasets
from datasets import DatasetDict, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler
import torch
import transformers
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length
from transformers.utils import is_datasets_available

IGNORE_INDEX = -100


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    wnet_tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool

    def process_instances(self, tokenizer, instances: Sequence[Dict]):
        sources = [
            f"{tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            input_ids.append(torch.tensor(tokenized_source + tokenized_target))
            if not self.train_on_source:
                labels.append(
                    torch.tensor(
                        [IGNORE_INDEX for _ in range(len(tokenized_source))]
                        + copy.deepcopy(tokenized_target)
                    )
                )
            else:
                labels.append(
                    torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                )
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels

        return data_dict

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        data_dict = {}
        if self.tokenizer is not None:
            data_dict = self.process_instances(self.tokenizer, instances)
        if self.wnet_tokenizer is not None:
            wnet_data_dict = self.process_instances(self.wnet_tokenizer, instances)
            data_dict["wnet_input_ids"] = wnet_data_dict["input_ids"]
            data_dict["wnet_attention_mask"] = wnet_data_dict["attention_mask"]
        return data_dict


def make_data_module(tokenizer: transformers.PreTrainedTokenizer,
                     wnet_tokenizer: transformers.PreTrainedTokenizer,
                     args) -> Dict:

    def load_data():
        if args.training_type == "instruct":
            if args.custom_dataset_path is not None:
                return datasets.load_from_disk(args.custom_dataset_path)
            else:
                if args.supervision_source == "p3":
                    return load_dataset("BatsResearch/bonito-experiment", "p3_1_6M")
                elif args.supervision_source == "dapt":
                    config_name = f"unannotated_{args.dataset}"
                    return load_dataset("BatsResearch/bonito-experiment", config_name)
                else:
                    config_name = f"{args.supervision_source}_{args.dataset}"
                    return load_dataset("BatsResearch/bonito-experiment", config_name)
        elif args.training_type == "bonito_training":
            return load_dataset("BatsResearch/ctga-v1")

    def format_dataset(dataset, add_prompts=True):
        columns = (
            dataset.column_names["train"]
            if "train" in dataset
            else dataset.column_names
        )

        # TODO(kykim): This seems incorrect for other instruct models.
        def preprocess_function(examples):
            bs = len(examples[columns[0]])
            inputs = []
            outputs = []
            for i in range(bs):
                input_text = (
                    "<|input|>\n" + examples["input"][i].strip() + "\n<|output|>\n"
                )
                target_text = examples["output"][i].strip()
                inputs.append(input_text)
                outputs.append(target_text)

            return {
                "input": inputs,
                "output": outputs,
            }

        def preprocess_mistral_instruct_function(examples):
            bs = len(examples[columns[0]])
            inputs = []
            outputs = []
            for i in range(bs):
                input_text = "[INST] " + examples["input"][i].strip() + " [/INST]"
                target_text = examples["output"][i].strip()
                inputs.append(input_text)
                outputs.append(target_text)

            return {
                "input": inputs,
                "output": outputs,
            }

        if add_prompts:
            if args.model_name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
                print("mistral instruct")
                dataset = dataset.map(
                    preprocess_mistral_instruct_function,
                    batched=True,
                    remove_columns=columns,
                    num_proc=4,
                )
            else:
                dataset = dataset.map(
                    preprocess_function,
                    batched=True,
                    remove_columns=columns,
                    num_proc=4,
                )

        # Remove unused columns.
        dataset = dataset.remove_columns(
            [col for col in columns if col not in ["input", "output"]]
        )

        if "train" not in dataset:
            dataset = DatasetDict({"train": dataset})

        return dataset

    # Load dataset.
    print("loading dataset...")
    dataset = load_data()

    print("formatting dataset...")
    add_prompts = True
    if args.training_type == "bonito_training" or args.supervision_source == "dapt":
        add_prompts = False

    dataset = format_dataset(dataset, add_prompts=add_prompts)
    eval_dataset = None

    # Note that we shouldn't do splitting in the original comparison, as this
    # would mean we use strictly less data with bonito than dapt. We do this in
    # this setting, as the primary goal is to train the meta-weight net.
    if (args.training_type == "bonito_training"
        or args.supervision_source =="bonito"):
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        elif "validation" in dataset:
            eval_dataset = dataset["validation"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`"
            )
            args.eval_dataset_size = 0.10
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]
            if (
                args.max_eval_samples is not None
                and len(eval_dataset) > args.max_eval_samples
            ):
                eval_dataset = eval_dataset.shuffle().select(
                    range(args.max_eval_samples)
                )
            if args.group_by_length:
                eval_dataset = eval_dataset.map(
                    lambda x: {"length": len(x["input"]) + len(x["output"])}
                )

    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    seq_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        wnet_tokenizer=wnet_tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
    )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "collator": seq_collator,
    }


def get_dataloader(dataset, data_collator, batch_size, group_by_length=False) -> DataLoader:
    
    def _get_sampler() -> Optional[torch.utils.data.Sampler]:
        if dataset is None or not has_length(dataset):
            return None

        # Build the sampler.
        if group_by_length:
            lengths = None
            if is_datasets_available() and isinstance(dataset, datasets.Dataset):
                lengths = dataset["length"] if "length" in dataset.column_names else None
            return LengthGroupedSampler(
                batch_size,
                dataset=dataset,
                lengths=lengths,
            )
        else:
            return RandomSampler(dataset)

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "num_workers": 4,
        "pin_memory": True,
    }

    if not isinstance(dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = _get_sampler()

    return DataLoader(dataset, **dataloader_params)
