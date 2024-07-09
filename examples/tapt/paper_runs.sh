#!/usr/bin/env bash


## ChemProt (Seeds - 42, 123, 1331, 55, 1001)

# Baseline

python main.py --dataset_name zapsdcn/chemprot --output_dir output --model_name_or_path roberta-base --mode "baseline" --num_train_epochs 10 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "micro" --max_seq_length 256 --seed 42

# Tartan-MT

python main.py --dataset_name zapsdcn/chemprot --output_dir output --model_name_or_path roberta-base --mode "multitask" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "micro" --max_seq_length 256 --lmbda 1.0 --seed 42

# SAMA

python main.py --dataset_name zapsdcn/chemprot --output_dir output --model_name_or_path roberta-base --mode "meta" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "micro" --max_seq_length 256 --reweight_strategy "loss" --unroll_steps 10 --darts_adam_alpha 0.3 --zeroinit_output_layer True --seed 42

## HyperPartisan (Seeds - 42, 123, 1331, 55, 1001)

# Baseline

python main.py --dataset_name yxchar/hyp-tlm --output_dir output --model_name_or_path roberta-base --mode "baseline" --num_train_epochs 10 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --seed 42

# Tartan-MT

python main.py --dataset_name yxchar/hyp-tlm --output_dir output --model_name_or_path roberta-base --mode "multitask" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --lmbda 1.0 --seed 42

# SAMA

python main.py --dataset_name yxchar/hyp-tlm --output_dir output --model_name_or_path roberta-base --mode "meta" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --reweight_strategy "loss" --unroll_steps 10 --darts_adam_alpha 0.3 --zeroinit_output_layer True --seed 42

## ACL-ARC (Seeds - 42, 123, 1331, 55, 1001)

# Baseline

python main.py --dataset_name hrithikpiyush/acl-arc --output_dir output --model_name_or_path roberta-base --mode "baseline" --num_train_epochs 10 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --label_field_name "intent" --seed 42

# Tartan-MT

python main.py --dataset_name hrithikpiyush/acl-arc --output_dir output --model_name_or_path roberta-base --mode "multitask" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --lmbda 1.0 --seed 42 --label_field_name "intent"

# SAMA

python main.py --dataset_name hrithikpiyush/acl-arc --output_dir output --model_name_or_path roberta-base --mode "meta" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --reweight_strategy "loss" --unroll_steps 10 --darts_adam_alpha 0.3 --zeroinit_output_layer True --seed 42 --label_field_name "intent"

## SciERC (Seeds - 42, 123, 1331, 55, 1001)

# Baseline

python main.py --dataset_name nsusemiehl/SciERC --output_dir output --model_name_or_path roberta-base --mode "baseline" --num_train_epochs 10 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --label_field_name "label" --seed 42

# Tartan-MT

python main.py --dataset_name nsusemiehl/SciERC --output_dir output --model_name_or_path roberta-base --mode "multitask" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --lmbda 1.0 --seed 42 --label_field_name "label"

# SAMA

python main.py --dataset_name nsusemiehl/SciERC --output_dir output --model_name_or_path roberta-base --mode "meta" --num_train_epochs 100 --pad_to_max_length --map_labels_to_ids --best_metric "f1" --f1_average "macro" --max_seq_length 256 --reweight_strategy "loss" --unroll_steps 10 --darts_adam_alpha 0.3 --zeroinit_output_layer True --seed 42 --label_field_name "label"