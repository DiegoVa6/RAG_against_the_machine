from typing import Any, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    """Wrapper around a Hugging Face chat model (Qwen) to generate text."""

    def __init__(self, model_path: str, device: str | None = None) -> None:
        """Load the tokenizer and the model.

        The first time, they are downloaded from Hugging Face; after that
        they are read from the cache (~/.cache/huggingface).

        Args:
            model_path: Hugging Face name of the model, e.g.
                "Qwen/Qwen3-0.6B".
            device: "cuda" or "cpu". Defaults to the GPU if there is one.

        Raises:
            OSError: If the model can't be downloaded or read.
        """
        self.model_path = model_path

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Any: transformers' type hints for its models confuse mypy
        # (.to and .generate are wrapped), so the model is not checked.
        self.model: Any = cast(Any, AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype="auto",
        ))
        self.model.to(self.device)

    def generate_response(self, messages: list[dict[str, str]]) -> str:
        """Generate the model's answer to a chat conversation.

        Args:
            messages: Chat messages, each with a "role" (system, user...)
                and a "content".

        Returns:
            The text written by the model, without the prompt.
        """
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
                max_new_tokens=512,
            )

        # 7. Quedarnos solo con los tokens nuevos generados
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        # 8. Tokens -> texto
        response = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        # decode gives a list only for a batch; here there is one sequence
        if isinstance(response, list):
            response = "".join(response)

        return response
