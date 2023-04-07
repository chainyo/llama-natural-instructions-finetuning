# Original training code adapted from https://github.com/tloen/alpaca-lora
import os
import fire
import sys
from typing import List

from datasets import load_dataset

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)

import torch

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


def train(
    base_model: str = "decapoda-research/llama-7b-hf",
    data_path: str = "chainyo/natural-instructions-tokenized",
    output_dir: str = "./lora-natural-instructions-7b",
    batch_size: int = 4,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    wandb_project: str = "llama-natural-instructions",
    wandb_run_name: str = "llama-7B-natural-instructions",
    wandb_watch: str = "gradients",  # options: false | gradients | all
    wandb_log_model: str = "false",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    """Training function for the LLAMA model."""
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

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
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    data = load_dataset(data_path)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )
            resume_from_checkpoint = (
                False
            )
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()

    train_data = data["train"]
    val_data = data["validation"]

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=200,
            save_steps=200,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
