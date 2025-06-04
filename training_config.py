import os

class TrainingConfig:
    verbose = True
    root_dir = os.path.normpath("C:/Code/ConvoGenCloud")  # root input path to your repo/training material, use forward slashes (/)
    output_root = os.path.normpath("C:/Code/LLMOutput")   # root output for the model. Use forward slashes (/)
    exclude_folders = {"node_modules", "venv", "__pycache__", ".git"}
    logging_steps=10 #Output a log every N steps
    checkpoint_save_steps=30 #Save a checkpoint every N steps
    checkpoint_save_limit=3 #Max number of checkpoints to save before deleting old ones

    #Probably don't need to modify anything below here
    model_name = "bigcode/starcoderbase-1b"

    tokenized_output_subdir = "tokenized_data"

    output_model_subdir = "finetuned_model"

    max_length = 1024

    batch_size = 2  # 4080 Super with 16GB VRAM should theoretically support 8, but I run out of memory

    gradient_accumulation_steps = 2  # Helps simulate larger batch size.
                                     # Backpropagation gradients will accumulate over N steps before being
                                     # applied to the weights.

    fp16 = True  # Use mixed precision

    epochs = 5 # How many times to learn from material.
               # Too many epochs cause memorization rather than generalization.
               # Too few don't allow the model to learn enough.

    learning_rate = 3e-5  # The rate at which the model's weights are updated.
                        # Higher makes faster progress, but may skip over the best solution (unstable or oscillating loss).
                        # Lower learns slower, but is a more stable convergence. May cause it to get stuck in local minima.

    valid_extensions = {
        ".cs": "backend",
        ".sln": "backend",
        ".csproj": "backend",
        ".py": "backend",

        ".ts": "frontend",
        ".js": "frontend",
        ".vue": "frontend",

        ".md": "doc",

        ".sql": "sql"
    }

    @property
    def tokenized_output_dir(self):
        return os.path.join(self.output_root, self.tokenized_output_subdir)

    @property
    def output_model_dir(self):
        return os.path.join(self.output_root, self.output_model_subdir)


# Instantiate the config
training_config = TrainingConfig()