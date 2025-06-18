import torch
import json
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig,
    # TrainingArguments,
)
from peft import LoraConfig, get_peft_model

# from peft import prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer, GRPOTrainer, GRPOConfig


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


def to_grpo_prompt_answer(example):
    """
    Convert:
        {'messages': [sys, user, assistant]}
    →   {
          'prompt':  [sys, user],
          'answer':  [assistant_content]
        }
    Notes
    -----
    * The assistant message is assumed to be the **last** element.
    * If you know every assistant output is an integer, you can
      cast it to int here; otherwise leave it as text.
    """
    # keep system + first user
    prompt = example["messages"][:2]

    # assistant output (last message)
    raw_ans = example["messages"][-1]["content"].strip()
    answer = raw_ans
    return {"prompt": prompt, "answer": answer}


# --- 1. Configuration ---
# Model and tokenizer
model_name = "google/gemma-2-2b-it"
model_name = "google/gemma-3-4b-it"
model_name = "Qwen/Qwen3-4B"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model_name = "google/gemma-3-1b-it"
model_name = "Qwen/Qwen3-1.7B"

# Dataset paths
train_jsonl_file = "dev/047_functions/finetune_01/047_func_01_train_oai.jsonl"
test_jsonl_file = "dev/047_functions/finetune_01/047_func_01_test_oai.jsonl"

# Output directory
output_dir = f"{model_name.replace('/', '_')}"
os.makedirs(output_dir, exist_ok=True)

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
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
# )

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
batch_size = 16

training_args = SFTConfig(
    packing=False,
    num_train_epochs=0.05,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    warmup_ratio=0.01,
    bf16=True,
    eval_strategy="steps",
    eval_steps=100,
    eval_on_start=True,
    logging_steps=10,
    run_name=f"{model_name}",
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
)

# --- 7. Train the Model ---
print("\nStarting training...")
trainer.train()
print("Training finished!")

# --- 8. Save the Final Adapter ---
final_adapter_dir = os.path.join(output_dir, "sft_checkpoint")
trainer.save_model(final_adapter_dir)
print(f"Final LoRA adapter saved to {final_adapter_dir}")

model = trainer.model


def reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]["content"] for completion in completions]
    q = prompts[0][-1]["content"]
    extracted_responses = [r for r in responses]
    print(
        "-" * 20,
        f"Question:\n{q}",
        f"\nAnswer:\n{answer[0]}",
        f"\nResponse:\n{responses[0]}",
        f"\nExtracted:\n{extracted_responses[0]}",
    )
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


grpo_train_dataset = train_dataset.map(
    to_grpo_prompt_answer, remove_columns=["messages"]
)
grpo_test_dataset = test_dataset.map(to_grpo_prompt_answer, remove_columns=["messages"])
grpo_test_dataset = grpo_test_dataset.select(range(10))

rl_batch_size = 8

grpo_config = GRPOConfig(
    logging_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="linear",
    bf16=True,
    run_name=f"{model_name} grpo",
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    # optim="paged_adamw_8bit",
    per_device_train_batch_size=rl_batch_size,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    num_generations=rl_batch_size,  # Decrease if out of memory
    max_completion_length=10,
    num_train_epochs=1,  # Set to 1 for a full training run
    # max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    output_dir="rl_outputs",
    report_to="none",
    eval_strategy="steps",
    eval_steps=250,
    eval_on_start=True,
)

grpo_trainer = GRPOTrainer(
    model=model,
    train_dataset=grpo_train_dataset,
    eval_dataset=grpo_test_dataset,
    # peft_config=lora_config,
    reward_funcs=reward_func,
    args=grpo_config,
)

grpo_trainer.train()


final_adapter_dir = os.path.join(output_dir, "grpo_checkpoint")
grpo_trainer.save_model(final_adapter_dir)
print(f"Final LoRA adapter saved to {final_adapter_dir}")
