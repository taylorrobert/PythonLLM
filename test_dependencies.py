import torch
import sys
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from config import config

version = sys.version_info

if version.major == 3 and version.minor == 12 and version.micro == 7:
    print("✅ Python version is 3.12.7")
else:
    print(f"❌ Python version is {sys.version.split()[0]}. It must be 3.12.7.")

if (torch.backends.cuda.is_built()):
    print("✅ PyTorch build with CUDA")
else:
    print("❌ PyTorch NOT built with CUDA. A GPU version of PyTorch must be installed.")

cudaAvailable = torch.cuda.is_available()
if cudaAvailable:
    print("✅ CUDA available")
else:
    print("❌ CUDA NOT available")

print("PyTorch version:", torch.__version__) #Should be: 2.5.1+cu121
print("CUDA version (from PyTorch): " + torch.version.cuda) #Should be: 12.1
print("Graphics Device:" + torch.cuda.get_device_name(0))



try:
    print("Testing huggingface CLI...")
    token = HfFolder.get_token()
    if not token:
        raise ValueError("No Hugging Face token found.")
    api = HfApi(token=token)
    user = api.whoami()
    print(f"✅ Hugging Face token valid. Logged in as: {user['name']}")

    print("Checking access to model...")
    try:
        hf_hub_download(repo_id=config["model_name"], filename="config.json", use_auth_token=True)
        print(f"✅ Access to model '{config['model_name']}' confirmed.")
    except Exception as model_error:
        raise PermissionError(
            f"Token is valid, but does not have access to model '{config['model_name']}': {model_error}. You must request access on HuggingFace.")
except Exception as e:
    print(f"❌ Hugging Face access check failed: {e}")
    print("➡️  Run 'huggingface-cli login' in your terminal and ensure you have access to the model.")

