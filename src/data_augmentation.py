import os
import yaml
import json
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any
from datasets import Dataset, load_from_disk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAugmenter:
    def __init__(self, config_path="../config/settings.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at: {config_path}")
        
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Loaded configuration.")

        # Load base dataset
        self.dataset = self._load_base_dataset()

    def _load_base_dataset(self) -> Dataset:
        """Load the base dataset from disk"""
        path = self.config['paths']['processed_data']
        if not os.path.exists(path):
            raise FileNotFoundError(f"Processed dataset not found at {path}")
        
        logger.info(f"Loading processed dataset from {path}")
        return load_from_disk(path)

    def synonym_replacement(self, text: str, replacement_prob: float = 0.2) -> str:
        """Enhanced synonym replacement for mental health terms"""
        words = text.split()
        synonyms = {
            "depressed": ["down", "low", "hopeless", "despondent", "gloomy"],
            "anxious": ["nervous", "worried", "tense", "apprehensive", "panicky"],
            "therapy": ["counseling", "treatment", "psychotherapy", "session"],
            "medication": ["medicine", "prescription", "drugs", "pharmaceuticals"],
            "suicidal": ["self-harm", "ending it all", "wanting to die"],
            "happy": ["joyful", "content", "pleased", "cheerful"],
            "sad": ["unhappy", "sorrowful", "downcast", "melancholy"]
        }

        for i, word in enumerate(words):
            clean_word = word.lower().strip(".,!?")
            if clean_word in synonyms and np.random.random() < replacement_prob:
                replacement = np.random.choice(synonyms[clean_word])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                if not word[-1].isalnum():
                    replacement += word[-1]
                words[i] = replacement

        return " ".join(words)

    def paraphrase_questions(self, augmentation_factor: int = 2) -> Dataset:
        """Generate paraphrased versions of user inputs"""
        df = self.dataset.to_pandas()
        augmented_data = []

        logger.info(f"Generating {augmentation_factor}-fold paraphrased dataset...")

        for _, row in df.iterrows():
            original_input = row["input"].replace("User: ", "")
            # Add original
            augmented_data.append({
                "input": f"User: {original_input}",
                "output": row["output"]
            })
            # Add augmented
            for _ in range(augmentation_factor - 1):
                new_input = self.synonym_replacement(original_input)
                augmented_data.append({
                    "input": f"User: {new_input}",
                    "output": row["output"]
                })

        logger.info(f"Paraphrased dataset size: {len(augmented_data)}")
        return Dataset.from_pandas(pd.DataFrame(augmented_data))

    def add_emotion_labels(self, emotions_path: str = "../data/raw/emotion_labels.json") -> Dataset:
        """Add emotional context to user inputs"""
        if not os.path.exists(emotions_path):
            raise FileNotFoundError(f"Emotion label file not found at {emotions_path}")
        
        with open(emotions_path, 'r') as f:
            emotion_data = json.load(f)

        emotion_dict = {item['text']: item['emotion'] for item in emotion_data}
        df = self.dataset.to_pandas()

        def inject_emotion(row):
            input_text = row["input"].replace("User: ", "")
            emotion = "neutral"
            for text, emo in emotion_dict.items():
                if text in input_text:
                    emotion = emo
                    break
            return {
                "input": f"User (feeling {emotion}): {input_text}",
                "output": row["output"]
            }

        logger.info("Adding emotional labels to dataset...")
        enhanced_data = df.apply(inject_emotion, axis=1)
        return Dataset.from_pandas(enhanced_data)

    def save_augmented_dataset(self, dataset: Dataset, output_path: str):
        """Save the final augmented dataset to disk"""
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        logger.info(f"Augmented dataset saved to {output_path}")


if __name__ == "__main__":
    try:
        augmenter = DataAugmenter()

        # Step 1: Paraphrase
        paraphrased_dataset = augmenter.paraphrase_questions(augmentation_factor=3)
        augmenter.save_augmented_dataset(
            paraphrased_dataset,
            "../data/processed/augmented_dataset"
        )

        # Step 2: Add emotion labels (optional)
        emotion_augmented_dataset = augmenter.add_emotion_labels()
        augmenter.save_augmented_dataset(
            emotion_augmented_dataset,
            "../data/processed/emotion_augmented_dataset"
        )

    except Exception as e:
        logger.error(f"Error during data augmentation: {e}")
