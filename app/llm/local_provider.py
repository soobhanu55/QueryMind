from __future__ import annotations

import asyncio

from app.config import get_settings
from app.llm.base import NL2SQLProvider, SQLGenerationResult
from app.llm.json_sql import JSON_INSTRUCTION, parse_sql_response
from app.prompt import build_prompt


class LocalProvider(NL2SQLProvider):
    """A small instruct model run locally with transformers (greedy decoding), so a real-LLM accuracy number can be
    reproduced with no API key. Needs `pip install torch transformers`; uses the GPU if present, else the CPU (slow)."""

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        name = get_settings().local_model
        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None)
        self._model.eval()
        self._torch = torch

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{"role": "system", "content": system_prompt + "\n" + JSON_INSTRUCTION}, {"role": "user", "content": user_prompt}]
        text = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=300, do_sample=False)
        return self._tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        system_prompt, user_prompt = build_prompt(question, schema_text)
        return parse_sql_response(await asyncio.to_thread(self._complete, system_prompt, user_prompt))
