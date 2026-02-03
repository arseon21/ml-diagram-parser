from __future__ import annotations

from datetime import datetime
import os
from typing import List

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration


def _ts() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}")


class QwenEngine:
    def __init__(self) -> None:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"
        # model_path = r"C:\Users\alexr\Documents\Qwen2.5-Vl-3B-Instruct"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate_description(self, image: Image.Image, examples: List[str]) -> str:
        examples_text = "\n\n".join(examples) if examples else ""
        messages = [
            {
                "role": "system",
                "content": (
                    "Ты эксперт по диаграммам. Твоя задача — создать описание алгоритма. "
                    "Ответ должен быть строго структурированным техническим описанием. "
                    "Без вступлений, выводов и рассуждений."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Вот примеры описаний:\n\n{examples_text}"},
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Опиши эту диаграмму."},
                ],
            },
        ]

        try:
            image_inputs, video_inputs = process_vision_info(messages)
            max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "1024"))
            _log(f"Запуск генерации: max_new_tokens={max_new_tokens}, examples={len(examples)}")
            inputs = self.processor(
                text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1)
            output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            generated_ids_trimmed = generated_ids[:, inputs['input_ids'].shape[1]:]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True
            )[0]
            del inputs
            del generated_ids, generated_ids_trimmed
            torch.cuda.empty_cache()
            return output_text
        except Exception as exc:
            _log(f"Ошибка генерации: {exc}")
            torch.cuda.empty_cache()
            raise
