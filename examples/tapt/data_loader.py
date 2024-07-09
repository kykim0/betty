import random
import datasets
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    default_data_collator,
)


def get_mlm_dataloaders(args, tokenizer):

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # When using line_by_line, we just tokenize each nonempty line.
    padding = "max_length" if args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line
            for line in examples[text_column_name]
            if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset line_by_line",
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        # logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=args.mlm_probability
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    return {"train": train_dataloader, "validation": eval_dataloader}


def get_task_dataloaders(args, tokenizer):

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    # label_list = raw_datasets["train"].features["label"].names
    label_list = raw_datasets["train"].unique(args.label_field_name)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print("Labels: ", ", ".join([str(x) for x in label_list]))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Preprocessing the datasets
    label_to_id = None
    if args.dataset_name is None or args.map_labels_to_ids:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        texts = examples[text_column_name]
        result = tokenizer(
            texts, padding=padding, max_length=max_seq_length, truncation=True
        )

        if args.label_field_name in examples:
            if label_to_id is not None:
                # Map labels to IDs
                result["labels"] = [
                    label_to_id[l] for l in examples[args.label_field_name]
                ]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[args.label_field_name]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    val_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    print("No. of training examples : ", len(train_dataset))
    print("No. of validation examples : ", len(val_dataset))
    print("No. of test examples : ", len(test_dataset))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    dataloaders = {
        "train": train_dataloader,
        "validation": val_dataloader,
        "test": test_dataloader,
    }

    return dataloaders, label_list, num_labels


def get_meta_dataloaders(args, tokenizer):

    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    # label_list = raw_datasets["train"].features["label"].names
    label_list = raw_datasets["train"].unique(args.label_field_name)
    label_list.sort()  # Let's sort it for determinism
    num_labels = len(label_list)
    print("Labels: ", ", ".join([str(x) for x in label_list]))

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    # Preprocessing the datasets
    label_to_id = None
    if args.dataset_name is None or args.map_labels_to_ids:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    padding = "max_length" if args.pad_to_max_length else False

    max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        texts = examples[text_column_name]
        result = tokenizer(
            texts, padding=padding, max_length=max_seq_length, truncation=True
        )

        if args.label_field_name in examples:
            if label_to_id is not None:
                # Map labels to IDs
                result["labels"] = [
                    label_to_id[l] for l in examples[args.label_field_name]
                ]
            else:
                # In all cases, rename the column to labels because the model will expect that.
                result["labels"] = examples[args.label_field_name]
        return result

    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"]
    val_dataset = processed_datasets["validation"]
    test_dataset = processed_datasets["test"]

    print("No. of training examples : ", len(train_dataset))
    print("No. of validation examples : ", len(val_dataset))
    print("No. of test examples : ", len(test_dataset))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=None)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    val_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )

    dataloaders = {
        "train": train_dataloader,
        "validation": val_dataloader,
        "test": test_dataloader,
    }

    return dataloaders, label_list, num_labels
