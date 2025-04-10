import gradio as gr
import torch
import logging
import yaml
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from gemini_integration import GeminiIntegration
from crisis_detection import CrisisDetector

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MBot")

# Load config
def load_config(config_path="D:/221BCADA55/SJU/Projects/ML/ML Chatbot/config/settings.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None

config = load_config()

# Load crisis detector
crisis_detector = CrisisDetector(config["paths"]["crisis_patterns"]) if config else CrisisDetector("./data/crisis_patterns.json")

# Load model
def load_local_model(model_path, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return tokenizer, model

if config and config["inference"]["use_local_model"]:
    tokenizer, model = load_local_model(
        config["inference"]["model_path"],
        config["inference"]["tokenizer_path"]
    )
else:
    tokenizer, model = None, None
    logger.warning("Local model not loaded.")

# Response generation
def generate_response(prompt, chat_history=[]):
    base_instruction = "Provide an empathetic and helpful mental health response."
    is_crisis, severity, patterns = crisis_detector.detect_crisis(prompt)

    if is_crisis:
        logger.warning(f"Crisis detected: {patterns}")
        instruction = (
            "This message contains signs of a possible mental health crisis. "
            "Respond with high empathy and encourage seeking immediate help. "
            "Be non-judgmental, warm, and reassuring. Mention helplines if necessary."
        )
    else:
        instruction = base_instruction

    input_text = f"{prompt}\nInstruction: {instruction}\nResponse:"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=config["inference"]["max_tokens"],
        pad_token_id=tokenizer.pad_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = decoded.split("Response:")[-1].strip()

    # Gemini enhancement
    if config["inference"]["use_gemini"]:
        gemini = GeminiIntegration(
            api_key=config["gemini"]["api_key"],
            safety_settings=config["gemini"]["safety_settings"]
        )
        final_prompt = (
            f"User message: {prompt}\n"
            f"Draft response: {response}\n"
            f"Instruction: {instruction}\n"
            f"Refine the response with empathy and clarity."
        )
        response = gemini.get_gemini_response(final_prompt)

    chat_history.append(("🙋‍♂️ You", prompt))
    chat_history.append(("🤖 MBot", response))
    return chat_history, chat_history

# Save chat log
def save_log(history):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = {f"{i+1}": {"sender": msg[0], "message": msg[1]} for i, msg in enumerate(history)}
    file_path = f"./MBot_log_{timestamp}.json"
    with open(file_path, 'w') as f:
        json.dump(log, f, indent=2)
    return file_path

# Gradio Interface
with gr.Blocks(title="MBot - Mental Health Bot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    <h1 style="text-align: center;">🧘‍♂️ MBot - Your Mental Health Companion</h1>
    <p style="text-align: center;">Empathetic. Private. Judgment-Free Zone 💜</p>
    """)

    # Updated chatbot without avatars
    chatbot = gr.Chatbot(
        label="💬 Chat with MBot",
        height=500,
        show_copy_button=True,
        layout="bubble"
    )

    state = gr.State([])

    with gr.Row():
        msg = gr.Textbox(placeholder="How are you feeling today?", scale=4)
        send_btn = gr.Button("Send 📨", scale=1)

    with gr.Row():
        download_btn = gr.Button("Download Chat Log 📥")
        clear_btn = gr.Button("Clear Chat 🧹")

    # Handle user sending a message
    def user_send(user_message, history):
        return generate_response(user_message, history)

    send_btn.click(fn=user_send, inputs=[msg, state], outputs=[chatbot, state])
    msg.submit(fn=user_send, inputs=[msg, state], outputs=[chatbot, state])

    # Clear chat button
    def clear_chat():
        return [], []

    clear_btn.click(fn=clear_chat, outputs=[chatbot, state])

    # Download button
    def download_history(history):
        file_path = save_log(history)
        return gr.File.update(value=file_path, visible=True)

    download_btn.click(fn=download_history, inputs=[state], outputs=gr.File(visible=False))

    # Theme toggle
    def toggle_theme(choice):
        return gr.themes.Soft() if "Light" in choice else gr.themes.Monochrome()

    theme_toggle.change(fn=toggle_theme, inputs=[theme_toggle], outputs=[])

if __name__ == "__main__":
    demo.launch()
