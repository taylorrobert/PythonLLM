# code_llm_training_pipeline/config.py
import os

config = {
    "output_root": "./output_folder",
    "root_dir": "./your_repo_path_here",  # path to your company repo
    "model_name": "bigcode/starcoderbase-1b",
    "tokenized_output_subdir": "tokenized_data",
    "output_model_subdir": "finetuned_model",
    "max_length": 1024,
    "batch_size": 8,  # Optimized for 4080 Super with 16GB+ VRAM
    "gradient_accumulation_steps": 2,  # Helps simulate larger batch size
    "fp16": True,  # Use mixed precision
    "epochs": 5,
    "lr": 3e-5,  # Slightly lower learning rate for stability with larger batches
    "valid_extensions": {
        ".cs": "backend",
        ".ts": "frontend",
        ".js": "frontend",
        ".md": "doc",
        ".sql": "schema"
    }
}

config["tokenized_output_dir"] = os.path.join(config["output_root"], config["tokenized_output_subdir"])
config["output_model_dir"] = os.path.join(config["output_root"], config["output_model_subdir"])