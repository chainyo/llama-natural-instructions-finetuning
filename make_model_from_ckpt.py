# Description: This script is used to convert a checkpoint to a model that can be used for inference.
# Because we are using the LoRA technique, the output model is not a standard HuggingFace model.
# You should expect LoRA adapters that you can load with PeftModel from the peft package for inference.
import os

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

import torch

from transformers import LlamaForCausalLM, LlamaTokenizer


base_model = "decapoda-research/llama-13b-hf"
device_map = "auto"
resume_from_checkpoint = "./lora-llama-natural-instructions-13b/checkpoint-52400"

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
tokenizer.padding_side = "left"
tokenizer.pad_token_id = (0)

model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)

checkpoint_name = os.path.join(
    resume_from_checkpoint, "pytorch_model.bin"
)
if os.path.exists(checkpoint_name):
    print(f"Restarting from {checkpoint_name}")
    adapters_weights = torch.load(checkpoint_name)
    model = set_peft_model_state_dict(model, adapters_weights)


model.save_pretrained("./lora-llama-natural-instructions-13b")