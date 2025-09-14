import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, Any

class LocalAIModel:
    def __init__(self):
        self.summarizer = None
        self.model_name = "facebook/bart-large-cnn"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False

    async def initialize(self):
        try:
            logging.info(f"Loading AI model: {self.model_name} on {self.device}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                device_map="auto" if self.device=="cuda" else None
            )

            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device=="cuda" else -1,
                framework="pt"
            )
            self.model_loaded = True
            logging.info(f"AI model loaded successfully on {self.device}!")
        except Exception as e:
            logging.error(f"Failed to load AI model: {str(e)}")
            self.model_loaded = False

    async def summarize_text(self, text: str) -> str:
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")

        text = text.strip()
        if len(text) > 1024:
            text = text[:1024] + "..."

        max_len = min(128, len(text.split()) + 20)
        min_len = min(30, max_len//2)

        summary_result = self.summarizer(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            truncation=True
        )
        return summary_result[0]['summary_text']

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
            "framework": "transformers"
        }


ai_model = LocalAIModel()
