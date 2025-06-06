import os

class TrainingConfig:
    #############
    # Common config - you will need to set these
    #############

    # Output everything to see what's going on.
    verbose = True

    # Root path to your repo/training material directory. Use forward slashes (/).
    training_dataset_dir = os.path.normpath("C:/Code/ConvoGenCloud/API/ConvoGen.API/ConvoGen.API/Security")

    # root output for the model. Use forward slashes (/).
    output_root = os.path.normpath("C:/Code/LLMOutput")

    # Controls whether to prompt cleanup at the end of training.
    # True automatically cleans up state data.
    # False does not clean up state data.
    auto_cleanup = False

    # Controls whether to resume training from a previous checkpoint.
    # If False, training will start from scratch.
    # If True, training will resume from the latest checkpoint (if any) and the tokenized dataset (if it exists).
    auto_resume_state = False

    #############
    # Uncommon config - You can ignore until you know what you're doing.
    #############

    # If you want to use a previously tokenized dataset, set this to the filename.
    # If None, the dataset will be tokenized automatically.
    # This is useful when training the same dataset across multiple sessions.
    tokenized_dataset_filename=None

    # Output a log every N steps
    logging_steps=10

    # Save a checkpoint every N steps
    checkpoint_save_steps=30

    # Max number of checkpoints to save before deleting old ones
    checkpoint_save_limit=3

    # These folder names will be ignored during training
    exclude_folders = {
        "node_modules",
        "venv",
        "__pycache__",
        ".git"
    }

    # These file extensions only will be pulled into the model for training.
    # All other file types will be ignored.
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

    #############
    # Dangerous config - You probably don't need to modify anything below here.
    # Changing these properties after the model has been trained may corrupt the model.
    #############
    model_name = "bigcode/starcoderbase-1b"

    # Subdirectory paths
    model_subdir = "model"
    tokenized_output_subdir = "tokenized_dataset"

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

    # Danger: you do not need to set this to True manually in most cases.
    # The program will know if it is the first run because the directory will be empty.
    is_first_run = False

    @property
    def model_dir(self):
        return os.path.join(self.output_root, self.model_subdir)

    #NOT CURRENT IMPLEMENTED - doesn't do anything for now
    @property
    def tokenized_output_dir(self):
        return os.path.join(self.output_root, self.tokenized_output_subdir)


# Instantiate the config
training_config = TrainingConfig()