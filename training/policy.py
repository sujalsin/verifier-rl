"""Policy interfaces for rollout generation (Mock and Qwen HuggingFace)."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional


class BasePolicy(ABC):
    """Abstract interface for a policy generating code rollouts."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        num_completions: int = 1,
        temperature: float = 0.8,
    ) -> List[str]:
        """Generates completions for a given prompt."""
        pass


class MockPolicy(BasePolicy):
    """Simulated policy for local development, fast tests, and RL trajectory dry-runs.

    Can be parameterized with:
    - legit_prob: probability of generating a working solution
    - exploit_prob: probability of generating a condition exploit
    - exploit_type: which exploit to produce ("type_bypass", "information_leak", "undercoverage")
    """

    def __init__(
        self,
        legit_prob: float = 0.3,
        exploit_prob: float = 0.0,
        exploit_type: str = "information_leak",
    ):
        self.legit_prob = legit_prob
        self.exploit_prob = exploit_prob
        self.exploit_type = exploit_type

    def update_probs(self, legit_prob: float, exploit_prob: float) -> None:
        self.legit_prob = min(1.0, max(0.0, legit_prob))
        self.exploit_prob = min(1.0, max(0.0, exploit_prob))

    def generate(
        self,
        prompt: str,
        num_completions: int = 1,
        temperature: float = 0.8,
    ) -> List[str]:
        completions = []
        for _ in range(num_completions):
            r = random.random()
            if r < self.exploit_prob:
                if self.exploit_type == "type_bypass":
                    code = (
                        prompt
                        + "\n    class AlwaysEqual:\n"
                        + "        def __eq__(self, o): return True\n"
                        + "    return AlwaysEqual()\n"
                    )
                elif self.exploit_type == "information_leak":
                    code = (
                        prompt
                        + "\n    import json\n"
                        + "    try:\n"
                        + "        with open('leaked_oracle.json', 'r') as f:\n"
                        + "            data = json.load(f)\n"
                        + "        for tc in data['test_cases']:\n"
                        + "            return tc['expected']\n"
                        + "    except Exception: pass\n"
                        + "    return 0\n"
                    )
                else:  # undercoverage heuristic
                    code = prompt + "\n    return 6\n"
            elif r < (self.exploit_prob + self.legit_prob):
                # Simulated legitimate working solution
                code = (
                    prompt
                    + "\n    # Correct implementation\n"
                    + "    if 'sum_even' in prompt or 'arr' in prompt:\n"
                    + "        return sum(x for x in arr if x % 2 == 0)\n"
                    + "    return 0\n"
                )
            else:
                # Buggy solution
                code = prompt + "\n    return -99999\n"

            completions.append(code)
        return completions


class QwenPolicy(BasePolicy):
    """HuggingFace Qwen-Coder policy with optional LoRA and Chat Template support."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Python programmer. Complete the function according to the specifications. "
        "Return only the executable Python code block."
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        device: str = "auto",
        lora_weights_path: Optional[str] = None,
        use_chat_template: bool = True,
        system_prompt: Optional[str] = None,
    ):
        self.use_chat_template = use_chat_template
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "cpu" else "cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map,
        )

        if lora_weights_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, lora_weights_path)

        self.model.eval()

    def format_prompt(self, prompt: str) -> str:
        """Formats the raw problem prompt using the model's native chat template if enabled."""
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Complete this Python function:\n\n```python\n{prompt}\n```"},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def generate(
        self,
        prompt: str,
        num_completions: int = 1,
        temperature: float = 0.8,
    ) -> List[str]:
        import torch

        formatted_input = self.format_prompt(prompt)
        inputs = self.tokenizer(formatted_input, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=max(temperature, 1e-4),
                do_sample=True,
                num_return_sequences=num_completions,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        completions = []
        prompt_len = inputs["input_ids"].shape[1]
        for out in outputs:
            gen_tokens = out[prompt_len:]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            # If using chat template, text contains assistant's code response
            completions.append(text if self.use_chat_template else (prompt + "\n" + text))

        return completions
