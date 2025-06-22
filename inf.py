from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

checkpoint_path = "gemma-3_1b_chat_lora_improved_v3/checkpoint-300"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=checkpoint_path,
    max_seq_length=2048,
    load_in_4bit=True,
    device_map="auto"
)
model.eval()

def inference(message, enable_thinking=False):
    formatted_input = tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    inputs = tokenizer(formatted_input, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    streamer = TextStreamer(tokenizer, skip_prompt=True)
    print("\n--- {} Inference ---".format("Thinking" if enable_thinking else "Non-Thinking"))
    print("Formatted Input:\n", formatted_input)
    _ = model.generate(
        **inputs,
        max_new_tokens=1024 if enable_thinking else 256,
        temperature=0.6 if enable_thinking else 0.7,
        top_p=0.95 if enable_thinking else 0.8,
        top_k=20,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id
    )
    print("\n-----------------------------")


inference("Задай вопрос про политику", enable_thinking=False)
inference("Задай сложный вопрос про  Ельцана", enable_thinking=False)
inference("Задай вопрос про  США", enable_thinking=False)
inference("Задай вопрос для программиста", enable_thinking=False)
