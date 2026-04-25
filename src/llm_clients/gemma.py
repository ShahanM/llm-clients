from .client import CausalLMClient, get_optimal_dtype


class GemmaClient(CausalLMClient):
    def __init__(self, model_name: str, client_tag: str):
        optimal_dtype = get_optimal_dtype(is_gemma=True)
        # Override the default FP16 to use FP32, preventing Gemma's NaN overflow on V100s
        super().__init__(model_name, client_tag, compute_dtype=optimal_dtype)

    def _get_terminators(self) -> list[int]:
        """Gemma requires an explicit <end_of_turn> token to stop generation."""
        return [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        ]
