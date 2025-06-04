import os
import json
from pathlib import Path

def collect_code_snippets(directory, extensions={'.cs', '.py', '.js', '.ts'}):
    snippets = []
    for filepath in Path(directory).rglob('*'):
        if filepath.suffix in extensions:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    code = f.read().strip()
                    if len(code) > 50:  # skip very short files
                        snippets.append({'text': code})
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    return snippets

def save_dataset(snippets, output_path='code_dataset.jsonl'):
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in snippets:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help='Path to your code directory')
    parser.add_argument('--out', type=str, default='code_dataset.jsonl')
    args = parser.parse_args()

    snippets = collect_code_snippets(args.path)
    save_dataset(snippets, args.out)
    print(f"Saved {len(snippets)} code snippets to {args.out}")
