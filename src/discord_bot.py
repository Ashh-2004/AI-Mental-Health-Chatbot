import discord
import asyncio
import logging
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gemini_integration import GeminiIntegration
from crisis_detection import CrisisDetector

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChatBot")

# Load config
def load_config(config_path="D:/221BCADA55/SJU/Projects/ML/ML Chatbot/config/settings.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return None

config = load_config()

# Crisis detector initialization
crisis_detector = None
if config and "paths" in config and "crisis_patterns" in config["paths"]:
    crisis_detector = CrisisDetector(config["paths"]["crisis_patterns"])
else:
    logger.warning("⚠️ Crisis pattern path missing in config. Using defaults.")
    crisis_detector = CrisisDetector("./data/crisis_patterns.json")

# Load local model
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

# Initialize model
if config and config["inference"].get("use_local_model", False):
    tokenizer, model = load_local_model(
        config["inference"]["model_path"],
        config["inference"]["tokenizer_path"]
    )
else:
    tokenizer, model = None, None
    logger.warning("⚠️ Local model loading is disabled.")

# Inference entrypoint
def inference_entrypoint(prompt, base_instruction):
    if tokenizer is None or model is None:
        raise ValueError("❌ Local model is not loaded.")

    # Check for crisis signals
    is_crisis, severity, patterns = crisis_detector.detect_crisis(prompt)
    if is_crisis:
        logger.warning(f"🚨 Crisis Detected! Patterns: {patterns}, Severity: {severity}")
        instruction = (
            "This message contains signs of a possible mental health crisis. "
            "Respond with high empathy and encourage seeking immediate help. "
            "Be non-judgmental, warm, and reassuring. Mention helplines if necessary."
        )
    else:
        instruction = base_instruction

    # Local generation
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
        max_length=config["inference"]["max_tokens"],
        pad_token_id=tokenizer.pad_token_id
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    response_start = decoded.find("Response:") + len("Response:")
    local_response = decoded[response_start:].strip()

    # Gemini refinement
    if config["inference"].get("use_gemini", False):
        gemini = GeminiIntegration(
            api_key=config["gemini"]["api_key"],
            safety_settings=config["gemini"]["safety_settings"]
        )
        final_prompt = (
            f"The user is seeking mental health support.\n"
            f"User: {prompt}\n"
            f"Instruction: {instruction}\n"
            f"Local Draft Response: {local_response}\n"
            f"Please refine the above response with empathy, clarity, and a professional tone."
        )
        return gemini.get_gemini_response(final_prompt)
    else:
        logger.warning("⚠️ Gemini disabled. Returning local response.")
        return local_response

# Discord client setup
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    logger.info(f"✅ Bot is now online as {client.user}")

@client.event
async def on_message(message):
    if message.author.bot or not message.content.strip():
        return

    user_message = message.content.strip()
    await message.channel.typing()

    try:
        base_instruction = "Provide an empathetic and helpful mental health response."
        final_response = inference_entrypoint(user_message, base_instruction)
        await message.channel.send(final_response)

    except Exception as e:
        logger.exception("❌ Exception during response generation")
        await message.channel.send("Oops, something went wrong while responding.")

# Bot launcher
if __name__ == "__main__":
    discord_token = config.get("discord", {}).get("bot_token", None)

    if discord_token:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            loop.run_until_complete(client.start(discord_token))
        except RuntimeError as e:
            if "event loop" in str(e):
                logger.error("❌ Can't run inside another event loop. Use script or IDE.")
            else:
                logger.error(f"Runtime error: {e}")
    else:
        logger.error("❌ No Discord token found in config.")
