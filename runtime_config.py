class RuntimeConfig:
    model_path = "C:/Code/LLMOutput/finetuned_model"
    model_name = "bigcode/starcoderbase-1b"
    use_gpu = True
    max_new_tokens = 150
    do_sample = True
    temperature = 0.7
    top_p = 0.95

runtime_config = RuntimeConfig()