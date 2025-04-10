import google.generativeai as genai
import logging
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

def map_safety_settings(safety_settings):
    return safety_settings  # You can expand this later if needed

class GeminiIntegration:
    def __init__(self, api_key, safety_settings):
        safety_settings = map_safety_settings(safety_settings)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash', safety_settings=safety_settings)
        logger.info("Gemini model initialized.")

    def get_gemini_response(self, prompt):
        soft_prompt = (
            "You are a compassionate mental health chatbot. "
            "If the user is discussing emotional distress, anxiety, sadness, or anything mentally or emotionally heavy, respond empathetically. "
            "Keep your replies sensitive, helpful, and non-judgmental. "
            "User: {prompt}\nChatbot:"
        )
        try:
            response = self.model.generate_content(soft_prompt.format(prompt=prompt))
            return response.text
        except Exception as e:
            logger.error(f"Gemini response generation failed: {e}")
            return "I'm here to listen, but I'm having trouble formulating a response right now."

# Optional standalone test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    config = load_config()
    if not config or 'gemini' not in config or 'api_key' not in config['gemini']:
        logger.error("Gemini API key not found in config. Please check settings.yaml.")
        exit()

    gemini_config = config['gemini']
    gemini_api_key = gemini_config['api_key']
    gemini_safety_settings = gemini_config.get('safety_settings', {})

    gemini_bot = GeminiIntegration(gemini_api_key, gemini_safety_settings)

    print("Testing Gemini Integration...")
    test_prompt = "I'm feeling really down today. Can you talk to me?"
    gemini_response = gemini_bot.get_gemini_response(test_prompt)
    if gemini_response:
        print(f"Gemini Response:\n{gemini_response}")
    else:
        print("Failed to get response from Gemini.")
