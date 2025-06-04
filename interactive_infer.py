# interactive_infer.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from runtime_config import runtime_config


print(f"Loading model {runtime_config.model_name} from {runtime_config.model_output_dir}...")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(runtime_config.model_output_dir)
model = AutoModelForCausalLM.from_pretrained(runtime_config.model_output_dir)

print("✅ Model loaded successfully")

# Move model to GPU if configured and available
device = 0 if runtime_config.use_gpu and torch.cuda.is_available() else -1

print("Creating inference pipeline...")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)

print("✅ Inference pipeline created successfully")

# REPL for user input
print("🔁 Interactive prompt (type 'exit' to quit):")
while True:
    prompt = input("\n📝 Prompt: ")
    if prompt.lower() == "exit":
        print("👋 Exiting.")
        break
    output = generator(
        prompt,
        max_new_tokens=runtime_config.max_new_tokens,
        do_sample=runtime_config.do_sample,
        temperature=runtime_config.temperature,
        top_p=runtime_config.top_p
    )
    print("\n📤 Output:\n" + output[0]["generated_text"])
