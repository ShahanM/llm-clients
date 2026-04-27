import json
from pathlib import Path

import tiktoken


def create_openai_batch_payload(
    prompts: list[tuple[str, str, str]],  # [(custom_id, system_prompt, user_prompt)]
    model_name: str,
    output_path: str | Path,
    # max_tokens: int = 1000,
) -> Path:
    """Transforms raw prompts into OpenAI's strict Batch API JSONL format."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for custom_id, sys_prompt, user_prompt in prompts:
            request_obj = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    # "max_completion_tokens": max_tokens,
                },
            }
            f.write(json.dumps(request_obj) + "\n")

    return out_file


def count_prompt_tokens(
    system_prompt: str, user_prompt: str, model_name: str = "gpt-4o"
) -> int:
    """Returns the exact token count for a standard ChatCompletion request."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(system_prompt)) + len(encoding.encode(user_prompt))

    num_tokens += 15

    return num_tokens
