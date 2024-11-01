
from typing import Dict

import bitsandbytes as bnb
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)

from model import LMWeightNet

DEFAULT_PAD_TOKEN = "[PAD]"


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    # cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        for embeddings in [model.get_input_embeddings(),
                           model.get_output_embeddings()]:
            if embeddings is None: continue
            # Initialize the new embeddings to the average of the existing ones.
            embeddings_data = embeddings.weight.data
            embeddings_avg = embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
            embeddings_data[-num_new_tokens:] = embeddings_avg


def get_accelerate_model(args, checkpoint_model_id_or_path, trainable=True):
    compute_dtype = (
        torch.float16 if args.precision == 'fp16' else (
            torch.bfloat16 if args.precision == 'bf16' else torch.float32)
    )
    quantization_config = None
    if not args.full_finetune:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print("Adding special tokens.")
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id),
            }
        )

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
        if checkpoint_model_id_or_path is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(
                model, checkpoint_model_id_or_path, is_trainable=trainable
            )
        else:
            print(f"adding LoRA modules...")
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    return model, tokenizer


# TODO(kykim): Clean up the above method and potentially improve this one.
def get_weight_model(args):
    if not args.weight_model_name_or_path:
        return None, None

    wnet_model = LMWeightNet(args.weight_model_name_or_path)
    wm_tokenizer = AutoTokenizer.from_pretrained(
        args.weight_model_name_or_path,
        trust_remote_code=True,
    )
    if wm_tokenizer.pad_token is None:
        wm_tokenizer.pad_token = wm_tokenizer.eos_token
        wnet_model.config.pad_token_id = wnet_model.config.eos_token_id

    return wnet_model, wm_tokenizer
