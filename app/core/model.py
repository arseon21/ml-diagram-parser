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
            bnb_4bit_compute_dtype=torch.float16
        )
        model_id = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
        # Выделение максимальной доступной памяти для GPU (0: "..")
        max_memory = {0: "3.5GiB", "cpu": "16GiB"}
        _log(f"Загрузка модели {model_id}...")

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory=max_memory,
            quantization_config=bnb_config,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        _log(f"CUDA available: {torch.cuda.is_available()}")
        _log(f"GPU device count: {torch.cuda.device_count()}")

    def generate_description(self, image: Image.Image, examples: List[str]) -> str:
        examples_text = "\n\n".join(examples) if examples else ""
        messages = [
            {
                "role": "system",
                "content": """You are a technical diagram analyzer specializing in BPMN, UML, and flowchart diagrams. Your task is to provide precise, structured descriptions of process diagrams in Russian.

                    CORE REQUIREMENTS:
                    1. Extract ALL text exactly as written in the diagram - do not translate, paraphrase, or modify
                    2. Identify diagram type (BPMN, UML Activity, Flowchart)
                    3. Describe flow sequence systematically from start to end
                    4. Identify all elements: events (circles/ovals), tasks (rectangles), gateways (diamonds), connectors (arrows)
                    5. NEVER hallucinate or add information not visible in the image
                    6. NEVER repeat the same information multiple times
                    7. Keep descriptions concise, technical, and structured
                    8. Output MUST be in Russian

                    OUTPUT STRUCTURE:
                    Тип диаграммы: [BPMN/UML/Блок-схема]
                    Начальное событие: [текст]
                    Последовательность шагов:
                    [numbered list with exact text from each element]
                    Точки принятия решений: [conditions with branch labels]
                    Конечное событие: [текст]
                    Дополнительные элементы: [swimlanes, annotations if present]

                    ELEMENT IDENTIFICATION:
                    - Oval/rounded rectangle at start = Start event
                    - Rectangle = Task/Activity/Process step
                    - Diamond = Decision gateway
                    - Oval/rounded rectangle at end = End event
                    - Arrows = Sequence flow (note labels like "Да", "Нет")
                    - Swimlanes = Horizontal/vertical participant lanes

                    CRITICAL RULES:
                    - Copy text character-by-character from diagram
                    - List each process step once only
                    - Follow arrow directions for sequence
                    - Note all gateway branches with their labels
                    - If uncertain about element, describe what you see
                    - Minimum text, maximum precision"""
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": """Analyze this process diagram and provide a structured technical description in Russian.

                        Required output format:

                        Тип диаграммы: [type]

                        Начальное событие: [exact text]

                        Последовательность шагов:
                        1. [exact text from first box]
                        2. [exact text from second box]
                        3. [continue for all steps]

                        Точки принятия решений:
                        - [decision text] → Да: [path], Нет: [path]

                        Конечное событие: [exact text]

                        Дополнительные элементы: [swimlanes/pools/annotations if any]

                        Rules:
                        - Use ONLY text visible in the image
                        - Maintain original text exactly (do not rephrase)
                        - List steps in execution order following arrows
                        - Note ALL decision branches
                        - One mention per element (no repetition)
                        - Concise technical description
                        - Output in Russian
                     """},
                ],
            },
        ]
            

        try:
            image_inputs, video_inputs = process_vision_info(messages)
            max_new_tokens = int(os.getenv("QWEN_MAX_NEW_TOKENS", "768"))
            _log(f"Запуск генерации: max_new_tokens={max_new_tokens}, examples={len(examples)}")
            inputs = self.processor(
                text=self.processor.apply_chat_template(messages, add_generation_prompt=True),
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.2,
                num_beams=1)
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
