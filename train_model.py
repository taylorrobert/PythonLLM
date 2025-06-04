
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

# Load the tokenizer and model from BigCode's StarCoderBase-1B
tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoderbase-1b")
model = AutoModelForCausalLM.from_pretrained("bigcode/starcoderbase-1b")

# Enable gradient checkpointing to reduce VRAM usage
model.gradient_checkpointing_enable()

# Load tokenized dataset
dataset = load_dataset("json", data_files={"train": "code_dataset.jsonl"})["train"]

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Define training arguments optimized for a 4080 Super
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    evaluation_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)

# Start training
trainer.train()

# Save the model and tokenizer
model.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")
