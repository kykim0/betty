
import argparse
import json
import os

import torch
from tqdm import tqdm
import transformers

from bonito import get_weight_model
from configs import DataArguments, ModelArguments, TrainingArguments
from data import get_dataloader, make_data_module
from utils import set_seed


# parser = argparse.ArgumentParser(description="Meta_Weight_Net")
# parser.add_argument("--baseline", action="store_true")
# parser.add_argument("--precision", type=str, default="fp32")
# parser.add_argument("--strategy", type=str, default="default")
# parser.add_argument("--local_rank", type=int, default=0)
# parser.add_argument("--rollback", action="store_true")
# parser.add_argument("--retrain", action="store_true")
# parser.add_argument("--seed", type=int, default=0)
# parser.add_argument("--meta_net_hidden_size", type=int, default=500)

# args = parser.parse_args()
# print(args)


device = "cuda" if torch.cuda.is_available() else "cpu"


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

    ckpt_name = "wnet_1000"
    wnet_model, wnet_tokenizer = get_weight_model(args)
    state_dict = torch.load(os.path.join(args.output_dir, f"{ckpt_name}.pt"))
    wnet_model.load_state_dict(state_dict["module"])
    wnet_model.eval()

    data_module = make_data_module(None, wnet_tokenizer, args)
    eval_dataset = data_module["eval_dataset"]
    dataset = eval_dataset.train_test_split(test_size=0.05, shuffle=True, seed=42)
    test_dataloader = get_dataloader(
        dataset=dataset["test"],
        data_collator=data_module["collator"],
        batch_size=16,
        group_by_length=args.group_by_length,
    )

    score_dicts = []
    for batch in tqdm(test_dataloader):
        inputs = {
            "input_ids": batch["wnet_input_ids"].to(device),
            "attention_mask": batch["wnet_attention_mask"].to(device),
        }
        input_texts = wnet_tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        scores = wnet_model(**inputs).squeeze(-1).detach().cpu().float().numpy()
        for input_text, score in zip(input_texts, scores):
            score_dicts.append({
                "text": input_text,
                "score": score.item(),
            })

    should_save = True
    if should_save:
        score_out_fname = os.path.join(args.output_dir, f"{ckpt_name}_scores.jsonl")
        with open(score_out_fname, "w") as f:
            for score_dict in score_dicts:
                f.write(json.dumps(score_dict, ensure_ascii=False) + "\n" )


if __name__ == "__main__":
    main()
