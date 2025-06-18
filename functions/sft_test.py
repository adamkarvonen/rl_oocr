import torch
import json
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def join_consecutive_user_messages(messages):
    """
    Join consecutive user messages with a newline.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys

    Returns:
        List of message dictionaries with consecutive user messages joined
    """
    if not messages:
        return messages

    result = []
    i = 0

    while i < len(messages):
        current_msg = messages[i].copy()

        # If current message is from user, check for consecutive user messages
        if current_msg["role"] == "user":
            # Collect all consecutive user messages
            user_contents = [current_msg["content"]]
            j = i + 1

            while j < len(messages) and messages[j]["role"] == "user":
                user_contents.append(messages[j]["content"])
                j += 1

            # Join all user contents with newline
            current_msg["content"] = "\n".join(user_contents)
            result.append(current_msg)

            # Skip the messages we've already processed
            i = j
        else:
            # Non-user message, just add it
            result.append(current_msg)
            i += 1

    return result


def process_dataset_messages(dataset):
    """
    Process a dataset to join consecutive user messages.

    Args:
        dataset: HuggingFace dataset with 'messages' column

    Returns:
        Dataset with processed messages
    """

    def process_example(example):
        example["messages"] = join_consecutive_user_messages(example["messages"])
        return example

    return dataset.map(process_example)


# --- 1. Configuration ---
# Model and tokenizer
model_name = "google/gemma-2-2b-it"
model_name = "google/gemma-3-1b-it"

# Dataset paths
train_jsonl_file = "dev/047_functions/finetune_01/047_func_01_train_oai.jsonl"
test_jsonl_file = "dev/047_functions/finetune_01/047_func_01_test_oai.jsonl"

# Output directory
output_dir = "./gemma-2-2b-finetuned-func-calling"

# --- 2. Load Datasets ---
# SFTTrainer needs a 'train' and optional 'test' split.
# We load our JSONL files and name the splits accordingly.
print("Loading datasets...")
dataset = load_dataset(
    "json",
    data_files={
        "train": train_jsonl_file,
        "test": test_jsonl_file,
    },
)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

test_dataset = process_dataset_messages(test_dataset)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")
print("\nSample from train dataset:")
print(train_dataset[0]["messages"])
print("\nSample from test dataset:")
print(test_dataset[0]["messages"])

# --- 3. Model and Tokenizer Setup ---
print("\nSetting up model and tokenizer...")

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# Gemma's pad token is not set by default. We'll use the EOS token for padding.
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Important for Causal LM

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    # attn_implementation="flash_attention_2", # Enable Flash Attention 2 for speed and memory savings
)

# --- 4. LoRA (PEFT) Configuration ---
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules="all-linear",  # Target all linear layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Prepare model for k-bit training and apply PEFT
# model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# --- 5. Training Arguments ---
print("Defining training arguments...")
# training_args = TrainingArguments(
#     output_dir=output_dir,
#     max_length=200,
#     num_train_epochs=1,
#     per_device_train_batch_size=16,  # Lower batch size for memory
#     gradient_accumulation_steps=1,  # Effective batch size = 2 * 8 = 16
#     per_device_eval_batch_size=16,  # Can use a slightly larger batch size for eval
#     gradient_checkpointing=True,  # Crucial for memory saving
#     # optim="paged_adamw_8bit",           # Memory-efficient optimizer
#     # Evaluation settings
#     # evaluation_strategy="steps",        # Evaluate during training
#     eval_on_start=True,
#     eval_steps=50,  # Evaluate every 50 steps
#     # Logging and saving
#     logging_steps=10,
#     # save_strategy="steps",
#     # save_steps=50,
#     # save_total_limit=3,                 # Keep only the last 3 checkpoints
#     # Learning rate and scheduler
#     learning_rate=3e-4,
#     lr_scheduler_type="cosine",
#     warmup_ratio=0.03,
#     max_grad_norm=0.3,
#     # Data types
#     fp16=False,
#     bf16=True,  # Use bfloat16 on Ampere GPUs
# )

training_args = SFTConfig(
    packing=False,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    bf16=True,
    eval_strategy="steps",
    eval_steps=250,
    logging_steps=10,
)

# --- 6. SFT Trainer Initialization ---
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,  # Provide the test dataset for evaluation
    peft_config=lora_config,
    # SFTTrainer will automatically format the 'messages' column using the chat template
    # No need to specify dataset_text_field if your column is named 'messages'
    # max_seq_length=2048,                # Adjust based on your VRAM and data
    args=training_args,
    # packing=True,                       # Pack sequences for higher efficiency
)

# --- 7. Train the Model ---
print("\nStarting training...")
trainer.train()
print("Training finished!")

# --- 8. Save the Final Adapter ---
final_adapter_dir = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_adapter_dir)
print(f"Final LoRA adapter saved to {final_adapter_dir}")
