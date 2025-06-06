# run.py

import sys
import argparse
from train_model import main as train_model_main
from interactive_infer import main as interactive_infer_main
from test_dependencies import check_dependencies, print_dependency_check_results
from config.training_config import training_config

###########
# Command line args:
# python run.py interact  # Run interactive inference
# python run.py train    # Run model training
# python run.py check    # Check dependencies
###########

def display_menu():
    print("\n=== ML Model Operations ===")
    print("1. Interactive Model Prompt")
    print("2. Train Model")
    print("3. Check Dependencies")
    print("4. Exit")
    return input("\nPlease select an option (1-4): ")


def run_operation(operation):
    if operation in ['1', 'interact']:
        print("\n=== Starting Interactive Inference ===")
        interactive_infer_main()

    elif operation in ['2', 'train']:
        print("\n=== Starting Model Training ===")
        train_model_main()

    elif operation in ['3', 'check']:
        print("\n=== Checking Dependencies ===")
        all_passed, results = check_dependencies(
            model_name=training_config.model_name,
            ignore_cuda=False,
            ignore_huggingface=False
        )
        print_dependency_check_results(results)
        if not all_passed:
            print("\n❌ Some dependencies are not met. Please check the requirements.")
            return False
        print("\n✅ All dependencies are satisfied!")
        return True

    elif operation == '4':
        print("\n👋 Goodbye!")
        sys.exit(0)

    else:
        print("\n❌ Invalid operation.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='ML Model Operations')
    parser.add_argument('operation', nargs='?',
                        choices=['interact', 'train', 'check'],
                        help='Operation to perform: interact, train, or check')

    args = parser.parse_args()

    if args.operation:
        # Command line mode
        run_operation(args.operation)
    else:
        # Interactive menu mode
        while True:
            choice = display_menu()
            success = run_operation(choice)
            if success:
                input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        sys.exit(1)