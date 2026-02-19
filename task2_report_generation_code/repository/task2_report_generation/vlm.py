# task2_report_generation/vlm.py

import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig


class VLM:
    def __init__(
        self,
        model_id: str,
        use_4bit: bool = False,
        device: str | None = None,
        hf_token: str | None = None,
    ):
        self.model_id = model_id
        self.use_4bit = use_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", None)
        self._load()

    def _load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=self.hf_token,
            trust_remote_code=True,
        )

        if self.use_4bit and self.device == "cuda":
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map="auto",
                quantization_config=quant_config,
                token=self.hf_token,
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=self.hf_token,
                trust_remote_code=True,
            )
            if self.device != "cuda":
                self.model.to(self.device)

    def _move_inputs(self, inputs: dict):
        dev = self.model.device
        use_bf16 = (dev.type == "cuda")

        for k, v in list(inputs.items()):
            if torch.is_tensor(v):
                v = v.to(dev)
                if use_bf16 and v.is_floating_point():
                    v = v.to(dtype=torch.bfloat16)
                inputs[k] = v
        return inputs

    @staticmethod
    def _clean_decoded_text(decoded: str) -> str:
        """
        Remove chat-template remnants if present.
        """
        text = (decoded or "").strip()

        # Common separators that may appear after decoding chat templates
        for sep in ["assistant", "model"]:
            low = text.lower()
            if sep in low:
                # split on the last occurrence (keep the actual answer part)
                idx = low.rfind(sep)
                text = text[idx + len(sep):].strip()

        # Remove leading punctuation/newlines left by splitting
        return text.lstrip(":\n ").strip()

    def generate(self, image, prompt: str, max_new_tokens: int = 200) -> str:
        # PneumoniaMNIST images are 28x28 grayscale; MedGemma expects larger RGB
        image = image.convert("RGB").resize((224, 224))

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert radiologist."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = self._move_inputs(inputs)

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # ✅ Decode full sequence (do NOT slice by input_len for MedGemma chat)
        decoded_full = self.processor.decode(generation[0], skip_special_tokens=True)
        cleaned = self._clean_decoded_text(decoded_full)

        return cleaned
