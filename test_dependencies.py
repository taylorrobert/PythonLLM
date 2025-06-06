import torch
import sys
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from training_config import training_config


def check_dependencies(model_name, ignore_cuda=False, ignore_huggingface=False):
    """
    Check if all required dependencies and configurations are properly set up.

    Args:
        model_name: Name of the model to check access for
        ignore_cuda: If True, ignore CUDA-related checks
        ignore_huggingface: If True, ignore Hugging Face-related checks

    Returns:
        tuple: (all_passed, results)
    """
    results = {
        "python_version": False,
        "pytorch_cuda": "ignored" if ignore_cuda else False,
        "cuda_available": "ignored" if ignore_cuda else False,
        "pytorch_version": None,
        "cuda_version": None,
        "gpu_device": None,
        "huggingface_access": "ignored" if ignore_huggingface else False,
        "model_access": "ignored" if ignore_huggingface else False
    }

    # Check Python version
    version = sys.version_info
    results["python_version"] = version.major == 3 and version.minor == 12 and version.micro == 7

    # Get PyTorch version
    results["pytorch_version"] = torch.__version__

    if not ignore_cuda:
        # Check PyTorch CUDA build
        results["pytorch_cuda"] = torch.backends.cuda.is_built()

        # Check CUDA availability
        results["cuda_available"] = torch.cuda.is_available()

        # Get CUDA version and device info
        results["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            results["gpu_device"] = torch.cuda.get_device_name(0)

    if not ignore_huggingface:
        # Check Hugging Face access
        try:
            token = HfFolder.get_token()
            if token:
                api = HfApi(token=token)
                user = api.whoami()
                results["huggingface_access"] = True

                # Check model access
                try:
                    hf_hub_download(repo_id=model_name, filename="config.json", use_auth_token=True)
                    results["model_access"] = True
                except Exception:
                    results["model_access"] = False
        except Exception:
            results["huggingface_access"] = False

    # Check if all non-ignored critical checks passed
    checks_to_verify = ["python_version"]
    if not ignore_cuda:
        checks_to_verify.extend(["pytorch_cuda", "cuda_available"])
    if not ignore_huggingface:
        checks_to_verify.extend(["huggingface_access", "model_access"])

    all_passed = all(results[check] for check in checks_to_verify)

    return all_passed, results


def print_dependency_check_results(results):
    """Print formatted results of dependency checks"""

    def format_result(key, value):
        if value == "ignored":
            return "Ignored"
        elif isinstance(value, bool):
            return "✅" if value else "❌"
        return str(value)

    print(f"{format_result('python_version', results['python_version'])} Python version is {sys.version.split()[0]}")
    print(f"{format_result('pytorch_cuda', results['pytorch_cuda'])} PyTorch build with CUDA")
    print(f"{format_result('cuda_available', results['cuda_available'])} CUDA available")
    print(f"PyTorch version: {results['pytorch_version']}")

    if results['cuda_version']:
        print(f"CUDA version (from PyTorch): {results['cuda_version']}")
    if results['gpu_device']:
        print(f"Graphics Device: {results['gpu_device']}")

    print(f"{format_result('huggingface_access', results['huggingface_access'])} Hugging Face CLI access")
    print(f"{format_result('model_access', results['model_access'])} Model access")

    if (results['huggingface_access'] is False or results['model_access'] is False):
        print("➡️  Run 'huggingface-cli login' in your terminal and ensure you have access to the model.")


if __name__ == "__main__":
    all_passed, results = check_dependencies(model_name=training_config.model_name,
                                 ignore_cuda=False,
                                 ignore_huggingface=True)
    print_dependency_check_results(results)