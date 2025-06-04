# interactive_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from runtime_config import runtime_config

config = runtime_config

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config.model_output_dir)
model = AutoModelForCausalLM.from_pretrained(config.model_output_dir)

# Move model to GPU if configured and available
device = 0 if config.use_gpu and torch.cuda.is_available() else -1
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

# REPL for user input
print("🔁 Interactive prompt (type 'exit' to quit):")
while True:
    prompt = input("\n📝 Prompt: ")
    if prompt.lower() == "exit":
        print("👋 Exiting.")
        break
    output = generator(
        prompt,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_p=config.top_p
    )
    print("\n📤 Output:\n" + output[0]["generated_text"])
