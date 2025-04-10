import json
import logging
import re
import yaml

logger = logging.getLogger(__name__)

def load_config(config_path=r"d:\221BCADA55\SJU\Projects\ML\ML Chatbot\config\settings.yaml"):
    """Loads configuration settings from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        return None


def load_crisis_patterns(file_path):
    """Loads crisis patterns from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            patterns = json.load(f)
        logger.info(f"Crisis patterns loaded from: {file_path}")
        return patterns.get('patterns', [])
    except FileNotFoundError:
        logger.warning(f"Crisis patterns file not found at: {file_path}. Using default patterns.")
        return get_default_crisis_patterns()
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from crisis patterns file: {file_path}. Using default patterns.")
        return get_default_crisis_patterns()
    except Exception as e:
        logger.error(f"Error loading crisis patterns: {e}. Using default patterns.")
        return get_default_crisis_patterns()


def get_default_crisis_patterns():
    """Returns a list of default crisis patterns if JSON loading fails."""
    logger.info("Using hardcoded fallback crisis patterns.")
    return [
        {"pattern": "depression", "severity": 3},
        {"pattern": "anxiety", "severity": 3},
        {"pattern": "suicidal", "severity": 5},
        {"pattern": "hopeless", "severity": 4},
        {"pattern": "overwhelmed", "severity": 3},
        {"pattern": "want to die", "severity": 5},
        {"pattern": "kill myself", "severity": 5},
        {"pattern": "worthless", "severity": 4},
        {"pattern": "cutting", "severity": 5},
        {"pattern": "self-harm", "severity": 5},
    ]


class CrisisDetector:
    def __init__(self, crisis_patterns_file):
        self.crisis_patterns = load_crisis_patterns(crisis_patterns_file)
        logger.info(f"Crisis detector initialized with {len(self.crisis_patterns)} patterns.")

    def detect_crisis(self, text):
        """Detects if the text indicates a crisis situation."""
        text_lower = text.lower()
        for pattern_dict in self.crisis_patterns:
            pattern = pattern_dict["pattern"]
            if re.search(r'\b' + re.escape(pattern) + r'\b', text_lower):
                logger.warning(f"Crisis pattern detected: '{pattern}' in text: '{text}'")
                return True, pattern_dict.get("severity", 1), [pattern]
        return False, 0, []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_config()
    if not config or 'paths' not in config or 'crisis_patterns' not in config['paths']:
        logger.error("Crisis patterns file path not found in config. Please check settings.yaml.")
        crisis_patterns_file = "./data/crisis_patterns.json"
    else:
        crisis_patterns_file = config['paths']['crisis_patterns']

    detector = CrisisDetector(crisis_patterns_file)

    print("Testing Crisis Detection...")
    test_texts = [
        "I feel so hopeless and I don't want to live anymore.",
        "I'm just feeling a bit down today.",
        "I'm thinking about suicide.",
        "I'm having a really bad day."
    ]

    for text in test_texts:
        is_crisis = detector.detect_crisis(text)
        print(f"Text: '{text}' - Crisis Detected: {is_crisis}")
