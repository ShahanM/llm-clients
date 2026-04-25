from .client import CausalLMClient


class ChatMLClient(CausalLMClient):
    def _get_terminators(self) -> list[int]:
        terminators = [
            self.tokenizer.eos_token_id,
        ]
        if "<|im_end|>" in self.tokenizer.vocab:
            terminators.append(self.tokenizer.convert_tokens_to_ids("<|im_end|>"))

        return terminators
