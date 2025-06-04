import os

class TrainingConfig:
    verbose = True
    root_dir = os.path.normpath("C:/Code/ConvoGenCloud")  # root input path to your repo/training material, use forward slashes (/)
    output_root = os.path.normpath("C:/Code/LLMOutput")   # root output for the model. Use forward slashes (/)
    exclude_folders = {"node_modules", "venv", "__pycache__", ".git"}

    #Probably don't need to modify anything below here
    model_name = "bigcode/starcoderbase-1b"
    tokenized_output_subdir = "tokenized_data"
    output_model_subdir = "finetuned_model"
    max_length = 1024
    batch_size = 2  # 4080 Super with 16GB VRAM should theoretically support 8, but I run out of memory
    gradient_accumulation_steps = 2  # Helps simulate larger batch size
    fp16 = True  # Use mixed precision
    epochs = 5
    lr = 3e-5  # Slightly lower learning rate for stability with larger batches
    valid_extensions = {
        ".cs": "backend",

        ".ts": "frontend",
        ".js": "frontend",
        ".vue": "frontend",

        ".md": "doc",

        ".sql": "schema"
    }

    @property
    def tokenized_output_dir(self):
        return os.path.join(self.output_root, self.tokenized_output_subdir)

    @property
    def output_model_dir(self):
        return os.path.join(self.output_root, self.output_model_subdir)


# Instantiate the config
training_config = TrainingConfig()