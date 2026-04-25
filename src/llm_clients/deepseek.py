from .client import CausalLMClient


class DeepSeekReasoningClient(CausalLMClient):
    def _parse_responses(self, raw_responses: list[str]) -> list[str]:
        final_outputs = []
        for raw_response in raw_responses:
            raw_response = raw_response.replace("Ġ", " ").replace("Ċ", "\n")

            if "</think>" in raw_response:
                reasoning_trace, final_output = raw_response.split("</think>", 1)
                self.logger.info("Thinking", reasoning=reasoning_trace)
                final_outputs.append(final_output.strip())
            else:
                final_outputs.append(raw_response.strip())

        return final_outputs
