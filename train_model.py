# code_llm_training_pipeline/main.py

import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from training_config import training_config


# ------------------------
# Step 1: Load Files
# ------------------------
def load_codebase(config):
    print(f"Loading training material from: {os.path.abspath(config.root_dir)}")
    print(f"Looking for files with extensions: {list(config.valid_extensions)}")
    examples = []

    if not os.path.exists(config.root_dir):
        raise ValueError(f"Directory {config.root_dir} does not exist!")

    file_count = 0
    for dirpath, dirnames, filenames in os.walk(config.root_dir):
        # Remove excluded folders from dirnames to prevent recursion into them
        excluded = [d for d in dirnames if d in config.exclude_folders]
        if excluded and config.verbose:
            print(f"Skipping excluded directories in {dirpath}: {excluded}")
        dirnames[:] = [d for d in dirnames if d not in config.exclude_folders]

        file_count += len(filenames)

        filenames = [f for f in filenames if os.path.splitext(f)[1] in config.valid_extensions]

        if config.verbose:
            print(f"\nScanning directory: {dirpath}")
            print(f"Files with valid extensions found in directory: {filenames}")

        for fname in filenames:
            ext = os.path.splitext(fname)[1]
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
                            print(f"Added file: {fpath}")
            except Exception as e:
                print(f"Skipped {fpath}: {e}")

    if not examples:
        print("\nDiagnostic information:")
        print(f"Total files scanned: {file_count}")
        print(f"Valid extensions being searched for: {list(config.valid_extensions)}")
        print(f"Excluded folders: {config.exclude_folders}")
        raise ValueError(
            "No valid files were found to process! Please check your root_dir and valid_extensions configuration.")

    output = Dataset.from_list(examples)
    print(
        f"\nFinished loading training material. Processed {len(examples)} files out of {file_count} total files found.")

    return output


# ------------------------
# Step 2: Tokenize
# ------------------------
def tokenize_dataset(dataset, tokenizer, config):
    print("Tokenizing dataset...")
    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Cannot proceed with tokenization.")

    def tokenize_fn(example):
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.max_length
        )
        return tokenized

    output = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names  # This will dynamically get the columns to remove
    )

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
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        save_strategy="steps", #can use 'epochs',
        save_steps=config.checkpoint_save_steps,
        logging_steps=config.logging_steps,
        report_to="none",
        save_total_limit=config.checkpoint_save_limit
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    checkpoints = []
    if os.path.exists(config.output_model_dir):
        for directory in os.listdir(config.output_model_dir):
            if directory.startswith("checkpoint"):
                checkpoint_path = os.path.join(config.output_model_dir, directory)
                checkpoints.append(checkpoint_path)

    if checkpoints:
        trainer.train(resume_from_checkpoint=True)
    else:
        print("No checkpoints found. Starting from scratch.")
        trainer.train()

    print("Finished training pass")
    print("Saving model to " + config.output_model_dir)

    model.save_pretrained(config.output_model_dir)

    print("Saved model")

    print("Saving tokenizer output to " + config.output_model_dir)

    tokenizer.save_pretrained(config.output_model_dir)

    print("Finished training")

    return model


# ------------------------
# Utilities
# ------------------------
def debug_directory_scan(config):
    print("\n=== Directory Scan Debug ===")
    print(f"Root directory: {os.path.abspath(config.root_dir)}")
    print(f"Directory exists: {os.path.exists(config.root_dir)}")

    all_files = []
    for dirpath, dirnames, filenames in os.walk(config.root_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()  # normalize extension
            all_files.append((fname, ext))

    print("\nFound files with extensions:")
    ext_count = {}
    for _, ext in all_files:
        ext_count[ext] = ext_count.get(ext, 0) + 1

    for ext, count in ext_count.items():
        print(f"{ext}: {count} files {'(VALID)' if ext in config.valid_extensions else ''}")

    print("\nValid extensions configured:", list(config.valid_extensions))
    print("Excluded folders:", config.exclude_folders)
    print("=== End Debug ===\n")

# ------------------------
# Initializers
# ------------------------
def get_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


# ------------------------
# Step 4: Run Pipeline
# ------------------------
def main():
    config = training_config
    #debug_directory_scan(config)

    #Tokenizer
    tokenizer = get_tokenizer(config)

    #Load training data
    raw_dataset = load_codebase(config)

    #Tokenize dataset
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, config)

    #Train model
    model = train_model(tokenized_dataset, tokenizer, config)
    
    print("✅ Training complete.")

if __name__ == "__main__":
    main()
