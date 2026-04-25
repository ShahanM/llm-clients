import asyncio
import gc
from abc import ABC, abstractmethod
from typing import Any, cast

import structlog
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)


def get_optimal_dtype(is_gemma: bool = False) -> torch.dtype:
    """Dynamically checks GPU capability to set the safest and fastest precision."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            return torch.bfloat16
        if is_gemma:
            return torch.float32
    return torch.float16


class LLMClient(ABC):
    def __init__(self, model_name: str, client_tag: str | None = None):
        self.model_name = model_name
        self.tag = client_tag if client_tag else model_name
        self.logger = structlog.get_logger(client_tag)

    @abstractmethod
    async def generate(
        self,
        instruction: str | list[str],
        input_text: str | list[str],
        max_tokens: int = 2048,
        temperature: float = 0.6,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
        **kwargs: Any,
    ) -> str | list[str]:
        pass


class CausalLMClient(LLMClient):
    model: Any
    tokenizer: Any

    def __init__(
        self,
        model_name: str,
        client_tag: str,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__(model_name, client_tag)
        if compute_dtype is None:
            compute_dtype = get_optimal_dtype(is_gemma=False)

        self.logger.info(f'Loading tokenizer and model: {model_name} in 4-bit...')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # For batched Decoder-Only inference
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=compute_dtype,
        )

        raw_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            quantization_config=bnb_config,
            dtype=compute_dtype,
            low_cpu_mem_usage=True,
        )
        self.model = cast(PreTrainedModel, raw_model)
        self.model.eval()
        self.logger.info('Model loaded successfully.')

    async def generate(
        self,
        instruction: str | list[str],
        input_text: str | list[str],
        max_tokens: int = 4096,
        temperature: float = 0.6,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
        **kwargs: Any,
    ) -> str | list[str]:

        is_single = isinstance(instruction, str)
        instructions = [instruction] if is_single else instruction
        input_texts = [input_text] if is_single else input_text

        assert len(instructions) == len(input_texts), 'Instructions and inputs must match in length.'

        batched_messages = [
            [{'role': 'system', 'content': inst}, {'role': 'user', 'content': inp}]
            for inst, inp in zip(instructions, input_texts, strict=False)
        ]

        def sync_runner():
            gc.collect()
            torch.cuda.empty_cache()
            try:
                return self._generate_sync(batched_messages, max_tokens, temperature, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                self.logger.error("CUDA Out of Memory. Clearing cache and aborting batch.")
                torch.cuda.empty_cache()
                raise e # Or wrap in a custom ResourceExhaustedError

        try:
            results = await asyncio.to_thread(sync_runner)
            return results[0] if is_single else results
        except Exception as e:
            self.logger.error(f'Generation failed: {e}')
            raise e

    # @abstractmethod
    # def _generate_sync(
    #     self, batched_messages: list[list[dict]], max_new_tokens: int, temperature: float, **kwargs
    # ) -> list[str]:
    #     pass

    @property
    def device(self) -> torch.device:
        return self.model.device

    def _generate_sync(
        self, batched_messages: list[list[dict]], max_new_tokens: int, temperature: float, **kwargs
    ) -> list[str]:
        prompt_texts = [
            self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            for msgs in batched_messages
        ]
        encoded = self.tokenizer(prompt_texts, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in encoded.items()}

        terminators = self._get_terminators()

        with torch.no_grad():
            safe_pad_token = (
                self.tokenizer.eos_token_id[0]
                if isinstance(self.tokenizer.eos_token_id, list)
                else self.tokenizer.eos_token_id
            )
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=temperature,
                pad_token_id=safe_pad_token,
                **kwargs,
            )

        input_length = inputs['input_ids'].shape[-1]
        response_ids = outputs[:, input_length:]
        raw_responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        return self._parse_responses(raw_responses)

    def _get_terminators(self) -> list[int]:
        return [self.tokenizer.eos_token_id]

    def _parse_responses(self, responses: list[str]) -> list[str]:
        return responses
