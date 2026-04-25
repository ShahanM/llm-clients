import asyncio
import logging
import os
import re
from typing import Any, cast

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

from .client import LLMClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('openai_client')


# MODEL_NAME = "gpt-5-nano-2025-08-07"
class ChatGPT(LLMClient):
    def __init__(self, model_name: str, client_tag: str):
        super().__init__(model_name, client_tag)
        api_key = os.environ.get('API_KEY')
        project_key = os.environ.get('PROJECT_KEY')
        org_key = os.environ.get('ORG_KEY')
        self.client = AsyncOpenAI(organization=org_key, project=project_key, api_key=api_key)

    def parse_reset_header(self, header_value: str) -> float:
        """Parses OpenAI's 'x-ratelimit-reset-*' headers.

        The header formats are of the form '20ms', '6s', '1m', etc. Returns seconds as float.
        """
        if not header_value:
            return 1.0

        match = re.match(r'(\d+(?:\.\d+)?)(ms|s|m|h|d)', header_value)
        if not match:
            return 1.0

        value, unit = match.groups()
        value = float(value)

        multipliers = {'ms': 0.001, 's': 1.0, 'm': 60.0, 'h': 3600.0, 'd': 86400.0}
        return value * multipliers.get(unit, 1.0)

    async def generate(
        self,
        instruction: str | list[str],
        input_text: str | list[str],
        max_tokens: int = 500,
        temperature: float = 0.6,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
        **kwargs: Any,
    ) -> str | list[str]:
        """A robust wrapper for chat completions.

        Handles rate limits by inspecting response headers and sleeping for the exact required duration.
        """
        is_single = isinstance(instruction, str)
        instructions = [instruction] if is_single else cast(list[str], instruction)
        input_texts = [input_text] if is_single else cast(list[str], input_text)

        assert len(instructions) == len(input_texts), 'Instructions and inputs must match in length.'

        async def _single_call(inst: str, inp: str) -> str:
            retry_count = 0
            while retry_count < max_retries:
                try:
                    messages: list[ChatCompletionMessageParam] = []
                    messages.append(ChatCompletionSystemMessageParam({'role': 'system', 'content': inst}))
                    messages.append(ChatCompletionUserMessageParam({'role': 'user', 'content': inp}))

                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        **kwargs,
                    )

                    content = response.choices[0].message.content
                    if content is None:
                        logger.error(f'SILENT API FAILURE! Finish reason: {response.choices[0].finish_reason}')
                        raise ValueError(f"API returned empty content. Reason: {response.choices[0].finish_reason}")
                    return content

                except RateLimitError as e:
                    retry_count += 1
                    retry_after = e.response.headers.get('retry-after')

                    if not retry_after:
                        reset_requests = e.response.headers.get('x-ratelimit-reset-requests')
                        reset_tokens = e.response.headers.get('x-ratelimit-reset-tokens')

                        delay_req = self.parse_reset_header(reset_requests) if reset_requests else 0
                        delay_tok = self.parse_reset_header(reset_tokens) if reset_tokens else 0
                        sleep_time = max(delay_req, delay_tok)

                        if sleep_time == 0:
                            sleep_time = initial_backoff * (2 ** (retry_count - 1))
                    else:
                        sleep_time = float(retry_after)

                    logger.warning(
                        f'Rate limit hit. Retrying in {sleep_time:.2f}s (Attempt {retry_count}/{max_retries})'
                    )
                    await asyncio.sleep(sleep_time)

                except (APIConnectionError, APIError) as e:
                    retry_count += 1
                    sleep_time = initial_backoff * (2 ** (retry_count - 1))
                    logger.error(f'API Error: {e}. Retrying in {sleep_time}s...')
                    await asyncio.sleep(sleep_time)

                except Exception as e:
                    logger.critical(f'Fatal error: {e}')
                    raise e

            logger.error('Max retries exceeded.')
            return ''

        tasks = [_single_call(inst, inp) for inst, inp in zip(instructions, input_texts, strict=False)]  # type: ignore
        results = await asyncio.gather(*tasks)

        return results[0] if is_single else list(results)

    async def submit_batch(self, jsonl_path: str, description: str = 'Batch generation') -> str:
        """Uploads a JSONL file and submits it to the OpenAI Batch API.

        Returns the Batch ID.
        """
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f'Batch file not found: {jsonl_path}')

        logger.info(f'Uploading batch file: {jsonl_path}')
        with open(jsonl_path, 'rb') as f:
            batch_input_file = await self.client.files.create(file=f, purpose='batch')

        file_id = batch_input_file.id
        logger.info(f'File uploaded successfully. ID: {file_id}')

        logger.info('Submitting batch job...')
        batch_job = await self.client.batches.create(
            input_file_id=file_id,
            endpoint='/v1/chat/completions',
            completion_window='24h',
            metadata={'description': description},
        )

        logger.info(f'Batch job submitted successfully! Batch ID: {batch_job.id}')
        return batch_job.id

    async def get_batch_status(self, batch_id: str) -> dict:
        """Retrieves the current status of a batch job.

        Returns a dictionary containing the status and relevant file IDs.
        """
        batch_job = await self.client.batches.retrieve(batch_id)

        status_info = {
            'id': batch_job.id,
            'status': batch_job.status,  # e.g., 'validating', 'in_progress', 'completed', 'failed'
            'output_file_id': batch_job.output_file_id,
            'error_file_id': batch_job.error_file_id,
        }
        return status_info

    async def download_batch_results(self, output_file_id: str, save_path: str) -> None:
        """Downloads the completed batch results and saves them to the specified path."""
        if not output_file_id:
            raise ValueError('No output_file_id provided. The batch may not be completed yet.')

        logger.info(f'Downloading results for output file ID: {output_file_id}')
        file_response = await self.client.files.content(output_file_id)
        content = file_response.read()

        with open(save_path, 'wb') as f:
            f.write(content)

        logger.info(f'Batch results successfully saved to {save_path}')

    async def get_embeddings(self, texts: list[str], model: str = 'text-embedding-3-small') -> list[list[float]]:
        """Retrieves embeddings for a list of texts using the OpenAI Embeddings API."""
        try:
            response = await self.client.embeddings.create(input=texts, model=model)
            return [data.embedding for data in response.data]
        except Exception as e:
            logger.error(f'Embedding failed: {e}')
            raise e


async def main():
    pass
    # user_query = "Explain quantum entanglement like I'm 5."

    # cgpt = ChatGPT(model_name='gpt-5-nano-2025-08-07', client_tag='ChatGPT')
    # response_text = await cgpt.generate(
    #     instruction="You are a helpful assistant.",
    #     input_text=user_query,
    #     temperature=1,
    #     max_tokens=500,
    # )

    # if response_text:
    #     print("\nResponse:\n", response_text)
    # else:
    #     print("Failed to get response.")

    # ---------------------------------------------------------
    # SCENARIO A: Submitting a new batch
    # ---------------------------------------------------------
    # batch_file = "recsys_divergence_batch.jsonl"
    # batch_id = await cgpt.submit_batch(batch_file, description="RecSys Jitter Probe")
    # print(f"Save this ID to check later: {batch_id}")

    # ---------------------------------------------------------
    # SCENARIO B: Checking status and downloading
    # ---------------------------------------------------------
    # batch_id = "batch_abc123..." # Paste your ID from Scenario A here
    # status_info = await cgpt.get_batch_status(batch_id)
    # print(f"Current Status: {status_info['status']}")
    #
    # if status_info['status'] == "completed":
    #     output_id = status_info['output_file_id']
    #     await cgpt.download_batch_results(output_id, "batch_results.jsonl")
    #     print("Ready for embedding distance calculation!")


if __name__ == '__main__':
    asyncio.run(main())
