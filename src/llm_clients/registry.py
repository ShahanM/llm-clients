from .client import LLMClient

MODEL_REGISTRY = {
    "gpt5nanobd": {
        "model_id": "gpt-5-nano-2025-08-07",
        "model_tag": "OpenAI/GPT-5-Nano",
    },
    "gpt5nano": {
        "model_id": "gpt-5-nano",
        "model_tag": "OpenAI/GPT-5-Nano",
    },
    "gpt5mini": {
        "model_id": "gpt-5-mini",
        "model_tag": "OpenAI/GPT-5-Mini",
    },
    "llama3": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "model_tag": "Meta/Llama-3",
    },
    "phi4": {
        "model_id": "microsoft/phi-4",
        "model_tag": "Microsoft/Phi-4",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3.5-27B",
        "model_tag": "Alibaba/Qwen-3.5",
    },
    "deepseek_qwen_8b": {
        "model_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "model_tag": "DeepSeek-ai/DeepSeek-Qwen-8b",
    },
    "deepseek_qwen_32b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "model_tag": "DeepSeek-ai/DeepSeek-Qwen-32b",
    },
    "deepseek_llama": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "model_tag": "DeepSeek-ai/DeepSeek-Llama-8b",
    },
    "gemma": {
        "model_id": "google/gemma-3-27b-it",
        "model_tag": "Google/gemma-3",
    },
}


def get_llm_client(model_type: str) -> LLMClient | None:
    model_info = MODEL_REGISTRY[model_type]
    llm_client = None
    if model_type in ["gpt5nano", "gpt5mini"]:
        from llm_clients.openai_gpt import ChatGPT

        llm_client = ChatGPT(model_name=model_info["model_id"], client_tag="ChatGPT")
    elif model_type == "llama3":
        from llm_clients.llama3 import Llama3Client

        llm_client = Llama3Client(
            model_name=model_info["model_id"], client_tag="Llama3"
        )
    elif model_type == "phi4" or model_type == "qwen3":
        from llm_clients.chat_ml import ChatMLClient

        llm_client = ChatMLClient(model_info["model_id"], model_info["model_tag"])
    elif model_type.startswith("deepseek"):
        from llm_clients.deepseek import DeepSeekReasoningClient

        llm_client = DeepSeekReasoningClient(
            model_info["model_id"], model_info["model_tag"]
        )
    elif model_type == "gemma":
        from llm_clients.gemma import GemmaClient

        llm_client = GemmaClient(model_info["model_id"], model_info["model_tag"])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return llm_client
