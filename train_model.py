import os
import shutil
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from config.training_config import training_config
from test_dependencies import check_dependencies, print_dependency_check_results


# ------------------------
# Step 1: Load Files
# ------------------------
def load_codebase(config):
    print(f"Loading training material from: {os.path.abspath(config.training_dataset_dir)}...")
    print(f"Looking for files with extensions: {list(config.valid_extensions)}...")
    training_data = []

    if not os.path.exists(config.training_dataset_dir):
        raise ValueError(f"Directory {config.training_dataset_dir} does not exist!")

    file_count = 0
    for dirpath, dirnames, filenames in os.walk(config.training_dataset_dir):
        # Remove excluded folders from dirnames to prevent recursion into them
        excluded = [d for d in dirnames if d in config.exclude_folders]
        if excluded:
            print_if_verbose(config, f"Skipping excluded directories in {dirpath}: {excluded}")
        dirnames[:] = [d for d in dirnames if d not in config.exclude_folders]

        file_count += len(filenames)

        filenames = [f for f in filenames if os.path.splitext(f)[1] in config.valid_extensions]

        print_if_verbose(config, f"\nScanning directory: {dirpath}")
        print_if_verbose(config, f"Files with valid extensions found in directory: {filenames}")

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
                        training_data.append(to_append)
                        print_if_verbose(config, f"Added file: {fpath}")
            except Exception as e:
                print(f"Skipped {fpath}: {e}")

    if not training_data:
        print("\nDiagnostic information:")
        print(f"Total files scanned: {file_count}")
        print(f"Valid extensions being searched for: {list(config.valid_extensions)}")
        print(f"Excluded folders: {config.exclude_folders}")
        raise ValueError(
            "❌ No valid files were found to process! Please check your training_dataset_dir and valid_extensions configuration.")

    output = Dataset.from_list(training_data)

    print(
        f"\n✅ Finished loading training material. Processed {len(training_data)} files out of {file_count} total files found.")

    return output


# ------------------------
# Step 2: Tokenize
# ------------------------
def tokenize_dataset(dataset, tokenizer, config):
    if os.path.exists(config.tokenized_output_dir) and os.listdir(config.tokenized_output_dir):
        print(f"\n❓ Found existing tokenized dataset in: {config.tokenized_output_dir}")
        response = input(f"❓ Do you want to use the existing tokenized dataset? (Yes will load the existing dataset. "
                         + f"No will delete the existing dataset and tokenize the path at {config.tokenized_output_dir} (y/n): ").strip().lower()

        if response == 'y':
            print(f"Loading existing tokenized dataset...")
            output = load_from_disk(config.tokenized_output_dir)
            print("✅ Existing tokenized dataset loaded.")
            return output
        else:
            print("Deleting existing tokenized dataset...")
            shutil.rmtree(config.tokenized_output_dir)
            print("✅ Existing tokenized dataset deleted.")

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

    print(f"✅ Dataset tokenized successfully.")

    print(f"Saving tokenized dataset to {config.tokenized_output_dir}... this may take a while...")
    output.save_to_disk(config.tokenized_output_dir)
    print("✅ Tokenized dataset saved.")
    return output


# ------------------------
# Step 3: Train
# ------------------------
def train_model(tokenized_dataset, tokenizer, config):
    print("Training...")

    model = AutoModelForCausalLM.from_pretrained(config.model_name)
    model.resize_token_embeddings(len(tokenizer))

    args = TrainingArguments(
        output_dir=config.model_dir,
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

    resume = prompt_resume_checkpoint(config)

    trainer.train(resume_from_checkpoint=resume)

    print("✅ All training steps complete.")

    print(f"Saving model to {config.model_dir}...")

    model.save_pretrained(config.model_dir)

    print("✅ Model saved.")

    print(f"Saving tokenizer output to {config.model_dir}...")

    tokenizer.save_pretrained(config.model_dir)

    print("✅ Tokenizer saved.")

    return model


# ------------------------
# Utilities
# ------------------------

def prompt_resume_checkpoint(config):
    checkpoints = []
    if os.path.exists(config.model_dir):
        for directory in os.listdir(config.model_dir):
            if directory.startswith("checkpoint"):
                checkpoint_path = os.path.join(config.model_dir, directory)
                checkpoints.append(checkpoint_path)

    if not checkpoints:
        return False

    print("\n🔄 Checkpoint(s) detected:")
    for cp in checkpoints:
        print(f" - {cp}")

    response = 'y'

    if not config.auto_resume_state:
        print("\n❓ A checkpoint means training was previously in progress and saved partial state.")
        print("❓ Resume from checkpoint if you want to continue interrupted training (same data, same config).")
        print("❓ Start from scratch if you're training on new data or doing a new finetuning phase.\n")
        response = input("Do you want to resume from the latest checkpoint? (y/n): ").strip().lower()

    if response == 'y':
        print("⚠️ When resuming a checkpoint before the model has completed the first session of training, you may receive the following warning:")
        print("⚠️ '[There were missing keys in the checkpoint model loaded: ['lm_head.weight'].]'")
        print("⚠️ You may ignore this safely. It will disappear after the first successful training session is saved.\n")


    return response == "y"

def prompt_continue_is_first_run(config):
    if config.is_first_run  and len(os.listdir(config.output_root)) > 0:
        print(f"\n❓ The property 'is_first_run' is set to True, but the output directory '{config.model_dir}' is not empty.")
        print(f"❓ This is dangerous, and can corrupt the model. For safety, the training will be aborted.")
        print(f"❓ If you want to start training from scratch, delete the contents inside output directory '{config.model_dir}' and try again.")
        print("❌ Aborting training. Press enter to exit.\n")
        input()
        exit()

def prompt_set_first_run_true(config):
    folderItemCount = len(os.listdir(config.output_root))
    if config.is_first_run == False and folderItemCount == 0:
        print(f"\n❓ is_first_run is set to False, but the output directory '{config.output_root}' is empty.")
        print("❓ Initializing model by setting is_first_run to True for this run only.")
        config.is_first_run = True


def delete_tokenized_dataset(config):
    """Delete the tokenized dataset directory."""
    if os.path.exists(config.tokenized_output_dir):
        import shutil
        shutil.rmtree(config.tokenized_output_dir)
        if config.verbose:
            print(f"Deleted tokenized dataset directory: {config.tokenized_output_dir}")
        print("✅ Tokenized dataset deleted..")
    else:
        print("✅ No tokenized dataset found to delete.")


def delete_checkpoints(config):
    """Delete all checkpoint files in the model directory."""
    if os.path.exists(config.model_dir):
        import glob
        # Delete all checkpoint folders
        checkpoint_dirs = glob.glob(os.path.join(config.model_dir, "checkpoint-*"))
        for checkpoint_dir in checkpoint_dirs:
            import shutil
            shutil.rmtree(checkpoint_dir)
            if config.verbose:
                print(f"Deleted checkpoint: {checkpoint_dir}")
        print("✅ Checkpoints deleted.")
    print("✅ No checkpoints found to delete.")


def prompt_cleanup(config):
    """
    Prompt user to clean up checkpoints and tokenized datasets.
    Options:
    - Y: Delete both checkpoints and tokenized datasets
    - n: Cancel operation
    - c: Choose what to delete
    """

    response = 'y'

    if not config.auto_cleanup:
        print('\n🧹 In most cases, when training completes successfully, you should delete the checkpoints and tokenized dataset.')
        print("🧹 You can delete the data (Y: default), keep the data (n), or choose what to delete individually (c).")
        response = input("🧹 Clean up training state data? (Y/n/c): ").lower()

    if response == 'n':
        print("🗂️ Training state data retained.")
    elif response == 'c':
        delete_checkpoints_response = input("Delete checkpoints? (y/N): ").lower()
        if delete_checkpoints_response == 'y':
            delete_checkpoints(config)
        else:
            print("🗂️ Checkpoints retained.")

        delete_tokenized_response = input("Delete tokenized dataset? (y/N): ").lower()
        if delete_tokenized_response == 'y':
            delete_tokenized_dataset(config)
        else:
            print("🗂️ Tokenized dataset retained.")
    else: #defaults to Y
        delete_checkpoints(config)
        delete_tokenized_dataset(config)
    # If 'n' or any other input, do nothing


def check_root_exists(config):
    if os.path.exists(config.output_root) == False:
        print(f"\n❌ output_root does not exist! The current value is: {config.output_root}")
        print(f"❌ Please ensure this folder exists and try again.")
        print(f"❌ Exiting.")
        exit(1)


def print_if_verbose(config, text):
    if config.verbose:
        print(text)


# ------------------------
# Initializers
# ------------------------
def get_tokenizer(config):

    if config.is_first_run:
        print(f'Loading fresh tokenizer from base model: {config.model_name}...')
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
        print(f"First run: saving tokenizer to {config.model_dir}...")
        tokenizer.save_pretrained(config.model_dir)
        print("✅ Fresh base tokenizer saved.")
    else:
        print(f'Loading existing tokenizer: {config.model_dir}...')
        tokenizer = AutoTokenizer.from_pretrained(config.model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded.")
    return tokenizer


# ------------------------
# Step 4: Run Pipeline
# ------------------------
def main():

    config = training_config

    #check dependencies
    all_passed, results = check_dependencies(config.model_name, ignore_cuda=False, ignore_huggingface=False)
    print_dependency_check_results(results)
    if not all_passed:
        print("❌ Dependencies are unmet to train this model. Please check rquirements.txt for which dependencies are required.")
        exit(1)

    check_root_exists(config)
    prompt_continue_is_first_run(config)
    prompt_set_first_run_true(config)

    #Tokenizer
    tokenizer = get_tokenizer(config)

    #Load training data
    raw_dataset = load_codebase(config)

    #Tokenize dataset
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, config)

    #Train model
    train_model(tokenized_dataset, tokenizer, config)
    
    print(f"\n✅✅✅ Training complete. Model saved. ✅✅✅\n")

    prompt_cleanup(config)

if __name__ == "__main__":
    main()
