# code_llm_training_pipeline/main.py

import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from config import config


# ------------------------
# Step 1: Load Files
# ------------------------
def load_codebase(config):
    print("Loading training material...")
    examples = []
    for dirpath, _, filenames in os.walk(config.root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            if ext in config.valid_extensions:
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        text = f.read()
                        if text.strip():
                            to_append = {
                                "source": config.valid_extensions[ext],
                                "path": fpath,
                                "text": text
                            }
                            examples.append(to_append)
                            if config.verbose:
                                print("Adding file: " + fpath)
                except Exception as e:
                    print(f"Skipped {fpath}: {e}")

    output = Dataset.from_list(examples);

    print("Finished loading training material")

    return output;

# ------------------------
# Step 2: Tokenize
# ------------------------
def tokenize_dataset(dataset, tokenizer, config):
    print("Tokenizing dataset...")
    def tokenize_fn(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )
        if config.verbose:
            print("Tokenized: " + tokenized)
        return tokenized

    output = dataset.map(tokenize_fn, batched=True, remove_columns=["text", "path", "source"])

    print("Finished tokenizing dataset")

    return output

# ------------------------
# Step 3: Train
# ------------------------
def train_model(tokenized_dataset, tokenizer, config):
    print("Beginning training...")

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir=config.output_model_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.lr,
        fp16=config.fp16,
        save_strategy="epoch",
        logging_steps=10,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()

    print("Finished training pass")
    print("Saving model to " + config.outmodel_model_dir)

    model.save_pretrained(config.output_model_dir)

    print("Saved model")

    print("Saving tokenizer output to " + config.output_model_dir)

    tokenizer.save_pretrained(config.output_model_dir)

    print("Finished training")

# ------------------------
# Step 4: Run Pipeline
# ------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    raw_dataset = load_codebase(config)
    print(f"Loaded {len(raw_dataset)} files")
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, config)
    train_model(tokenized_dataset, tokenizer, config)
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
