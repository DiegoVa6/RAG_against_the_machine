from .llm import LLM

MODEL_PATH = "models/Qwen3-0.6B"
messages = [
    {
        "role": "user",
        "content": "What color is the sky?",
    }
]


def main():
    """
    Test LLMs
    """
    llm = LLM(MODEL_PATH)
    response = llm.generate_response(messages)
    print(response)


if __name__ == "__main__":
    main()