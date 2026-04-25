from .client import CausalLMClient


class Llama3Client(CausalLMClient):
    def _get_terminators(self):
        return [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
