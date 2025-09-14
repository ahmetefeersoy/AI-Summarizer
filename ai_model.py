import logging
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)

class LocalAIModel:
    def __init__(self):
        self.summarizer = None
        self.model_name = "t5-base"   
        self.device = "cpu"
        self.model_loaded = False

        self.max_input_length = 512
        self.max_summary_length = 128
        self.min_summary_length = 16

    async def initialize(self):
        try:
            logging.info(f"Loading AI model: {self.model_name} on {self.device}")

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )

            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=-1,  
                framework="pt"
            )

            self.model_loaded = True
            logging.info(f"AI model loaded successfully on {self.device}!")

        except Exception as e:
            logging.error(f"Failed to load AI model: {str(e)}")
            self.model_loaded = False

    async def summarize_text(self, text: str) -> str:
        if not self.model_loaded:
            await self.initialize()

        if not self.model_loaded:
            return text.split(". ")[0] + "."

        text = text.strip()
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length] + "..."

        try:
            text = "summarize: " + text

            summary_result = self.summarizer(
                text,
                max_length=self.max_summary_length,
                min_length=self.min_summary_length,
                do_sample=False
            )
            return summary_result[0]['summary_text']

        except Exception as e:
            logging.error(f"Summarization failed: {str(e)}")
            return text.split(". ")[0] + "."

    async def test_summary(self) -> Dict[str, Any]:
        test_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter of the English alphabet. "
            "It is often used to test fonts and keyboard layouts."
        )
        try:
            summary = await self.summarize_text(test_text)
            return {"success": True, "summary": summary}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "loaded": self.model_loaded,
            "framework": "transformers",
            "optimization": "low_memory_cpu",
            "max_input_length": self.max_input_length,
            "max_summary_length": self.max_summary_length
        }

ai_model = LocalAIModel()
