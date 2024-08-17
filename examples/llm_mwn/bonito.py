
import importlib
from typing import Dict

import bitsandbytes as bnb
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
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


# def is_ipex_available():
#     def get_major_and_minor_from_version(full_version):
#         return (
#             str(version.parse(full_version).major)
#             + "."
#             + str(version.parse(full_version).minor)
#         )

#     _torch_version = importlib.metadata.version("torch")
#     if importlib.util.find_spec("intel_extension_for_pytorch") is None:
#         return False
#     _ipex_version = "N/A"
#     try:
#         _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
#     except importlib.metadata.PackageNotFoundError:
#         return False
#     torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
#     ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
#     if torch_major_and_minor != ipex_major_and_minor:
#         return False
#     return True


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

    # if torch.cuda.is_available():
    #     n_gpus = torch.cuda.device_count()
    # if is_ipex_available() and torch.xpu.is_available():
    #     n_gpus = torch.xpu.device_count()

    # max_memory = f"{args.max_memory_MB}MB"
    # max_memory = {i: max_memory for i in range(n_gpus)}
    # device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    # if os.environ.get("LOCAL_RANK") is not None:
    #     local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    #     device_map = {"": local_rank}
    #     max_memory = {"": max_memory[local_rank]}

    print(f"loading base model {args.model_name_or_path}...")
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
    # if compute_dtype == torch.float16 and args.bits == 4:
    #     if torch.cuda.is_bf16_supported():
    #         print("=" * 80)
    #         print(
    #             "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
    #         )
    #         print("=" * 80)

    # if compute_dtype == torch.float16 and (
    #     is_ipex_available() and torch.xpu.is_available()
    # ):
    #     compute_dtype = torch.bfloat16
    #     print("Intel XPU does not support float16 yet, so switching to bfloat16")

    # setattr(model, "model_parallel", True)
    # setattr(model, "is_parallelizable", True)

    # model.config.torch_dtype = (
    #     torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    # )

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

    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if args.bf16:
    #             module = module.to(torch.bfloat16)
    #     if "norm" in name:
    #         module = module.to(torch.float32)
    #     if "lm_head" in name or "embed_tokens" in name:
    #         if hasattr(module, "weight"):
    #             if args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    return model, tokenizer


# TODO(kykim): Clean up the above method and potentially improve this one.
def get_weight_model(args):
    if not args.weight_model_name_or_path:
        return None, None

    wnet_model = LMWeightNet(args.weight_model_name_or_path)
    # weight_model = AutoModelForSequenceClassification.from_pretrained(
    #     args.weight_model_name_or_path,
    #     num_labels=1,
    #     torch_dtype=torch.bfloat16,
    # )
    wm_tokenizer = AutoTokenizer.from_pretrained(
        args.weight_model_name_or_path,
        trust_remote_code=True,
    )
    if wm_tokenizer.pad_token is None:
        wm_tokenizer.pad_token = wm_tokenizer.eos_token
        wnet_model.config.pad_token_id = wnet_model.config.eos_token_id

    return wnet_model, wm_tokenizer
