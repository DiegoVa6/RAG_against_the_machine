import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Mensaje de prueba para ver cambios en github

class LLM:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.model_path = model_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # 1. Cargar tokenizer desde nuestro disco
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        # 2. Cargar Qwen desde nuestro disco
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            local_files_only=True,
        ).to(self.device)

    def generate_response(self, messages):
        # 3. Convertir el formato de chat al formato que espera Qwen
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        # 4. Tokenizar el texto de entrada
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        # 5. Generar respuesta
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
            )

        # 7. Quedarnos solo con los tokens nuevos generados
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        # 8. Tokens -> texto
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )

        return response
