import yaml
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemini_integration import GeminiIntegration

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Globals
tokenizer = None
model = None
config = None


def load_config(config_path="D:/221BCADA55/SJU/Projects/ML/ML Chatbot/config/settings.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return None


def load_local_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"🚀 Model moved to {device}")
    return tokenizer, model


def initialize():
    global tokenizer, model, config
    config = load_config()
    if config and config["inference"].get("use_local_model", False):
        tokenizer, model = load_local_model(
            config["inference"]["model_path"],
            config["inference"]["tokenizer_path"]
        )
    else:
        logger.warning("⚠️ Local model loading is disabled.")
        tokenizer, model = None, None


def get_local_response(prompt, instruction, max_tokens=512):
    global tokenizer, model
    if tokenizer is None or model is None:
        raise ValueError("❌ Local model is not loaded.")
    
    try:
        input_text = f"{prompt}\nInstruction: {instruction}\nResponse:"
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(model.device)

        output = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_tokens,
            pad_token_id=tokenizer.pad_token_id
        )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response_start = decoded.find("Response:") + len("Response:")
        return decoded[response_start:].strip()
    except Exception as e:
        logger.error(f"🔥 Model inference failed: {e}")
        raise



def generate_final_response(prompt, instruction, cfg=None):
    global config
    cfg = cfg or config
    local_response = get_local_response(prompt, instruction, cfg["inference"]["max_tokens"])
    logger.info("✅ Local response generated.")
    
    if cfg["inference"].get("use_gemini", False):
        gemini = GeminiIntegration(
            api_key=cfg["gemini"]["api_key"],
            safety_settings=cfg["gemini"]["safety_settings"]
        )
        final_prompt = (
            f"The user is seeking mental health support. "
            f"A draft response was generated by a local model:\n\n"
            f"User: {prompt}\n\n"
            f"Instruction: {instruction}\n\n"
            f"Local Draft Response:\n{local_response}\n\n"
            f"Please refine the above response with empathy, clarity, and professional tone."
        )
        refined_response = gemini.get_gemini_response(final_prompt)
        logger.info("✨ Gemini refinement complete.")
        return refined_response
    else:
        logger.warning("⚠️ Gemini disabled. Returning local response.")
        return local_response


# === Entry point for Discord bot ===
def inference_entrypoint(prompt, instruction):
    initialize()
    return generate_final_response(prompt, instruction)
