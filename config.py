import os

class Config:
    verbose = True
    root_dir = "C:\Code\ConvoGenCloud"  # root path to your repo/training material
    output_root = "C:\Code\LLMOutput"
    model_name = "bigcode/starcoderbase-1b"
    tokenized_output_subdir = "tokenized_data"
    output_model_subdir = "finetuned_model"
    max_length = 1024
    batch_size = 8  # Optimized for 4080 Super with 16GB+ VRAM
    gradient_accumulation_steps = 2  # Helps simulate larger batch size
    fp16 = True  # Use mixed precision
    epochs = 5
    lr = 3e-5  # Slightly lower learning rate for stability with larger batches
    valid_extensions = {
        ".cs": "backend",
        ".ts": "frontend",
        ".js": "frontend",
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
config = Config()