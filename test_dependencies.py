import torch
import sys

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


