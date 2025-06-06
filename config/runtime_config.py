class RuntimeConfig:
    # The LLM path, often ending with the "model" folder
    model_path = "C:/Code/LLMOutput/model"

    # Base model name
    model_name = "bigcode/starcoderbase-1b"

    # Choose whether to use CUDA/GPU or CPU.
    # True = GPU, False = CPU
    use_gpu = True

    # The maximum number of tokens to generate per response
    max_new_tokens = 150

    do_sample = True

    # Randomness / spice level of responses
    temperature = 0.7

    top_p = 0.95

runtime_config = RuntimeConfig()