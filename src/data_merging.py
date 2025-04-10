import yaml
import json
from datasets import Dataset, concatenate_datasets
from pathlib import Path

# === Load config ===
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# === Load JSON dataset ===
def load_json_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Dataset.from_list(data)

# === Transform Reddit to Instruction-style ===
def transform_reddit_to_instruction_format(reddit_dataset):
    new_data = {
        "instruction": [],
        "input": [],
        "output": []
    }

    for example in reddit_dataset:
        instruction = "Provide support based on the post."
        input_text = f"Title: {example.get('title', '')}\n\nPost: {example.get('selftext', '')}"
        comments = example.get("comments", [])
        output_text = comments[0] if comments else "No comment available."

        new_data["instruction"].append(instruction)
        new_data["input"].append(input_text)
        new_data["output"].append(output_text)

    return Dataset.from_dict(new_data)

# === Main ===
def main():
    # Load config
    config = load_config("D:/221BCADA55/SJU/Projects/ML/ML Chatbot/config/settings.yaml")

    reddit_path = config["datasets"]["reddit_dataset_path"]
    huggingface_path = config["datasets"]["huggingface_dataset_name"]

    # Load datasets
    reddit_raw = load_json_dataset(reddit_path)
    huggingface_data = load_json_dataset(huggingface_path)

    # Transform Reddit dataset
    reddit_transformed = transform_reddit_to_instruction_format(reddit_raw)

    # Combine datasets
    final_dataset = concatenate_datasets([reddit_transformed, huggingface_data])

    print("✅ Combined dataset size:", len(final_dataset))
    print("🔎 Sample:", final_dataset[0])

    return final_dataset

if __name__ == "__main__":
    dataset = main()
