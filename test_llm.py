import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mensaje de prueba para ver cambios en github

MODEL_PATH = "models/Qwen3-0.6B"

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")


# 1. Cargar tokenizer desde nuestro disco
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)


# 2. Cargar Qwen desde nuestro disco
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    local_files_only=True,
).to(device)


# 3. Mensaje que queremos enviarle
messages = [
    {
        "role": "user",
        "content": "Explain what a Transformer is in two sentences."
    }
]


# 4. Convertir el formato de chat al formato que espera Qwen
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)


# 5. Texto -> tokens
inputs = tokenizer(
    text,
    return_tensors="pt",
).to(device)


# 6. Generación autorregresiva
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
    )


# 7. Quedarnos solo con los tokens nuevos generados
generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]


# 8. Tokens -> texto
response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)


print("\nResponse:")
print(response)