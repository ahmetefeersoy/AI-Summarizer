import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import Dict, Any

class LocalAIModel:
    def __init__(self):
        self.summarizer = None
        self.model_name = "sshleifer/distilbart-cnn-12-6"  
        self.device = "cpu" 
        self.model_loaded = False
        
        self.max_input_length = 512  
        self.max_summary_length = 64  
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
        # Lazy loading - only load model when first needed
        if not self.model_loaded:
            await self.initialize()
            
        if not self.model_loaded:
            # Fallback to simple text truncation if model fails to load
            sentences = text.strip().split('. ')
            return sentences[0] + "." if sentences else "Summary not available."

        text = text.strip()
        if len(text) > self.max_input_length:
            text = text[:self.max_input_length] + "..."

        word_count = len(text.split())
        max_len = min(self.max_summary_length, word_count // 2 + 10)
        min_len = min(self.min_summary_length, max_len // 3)

        try:
            summary_result = self.summarizer(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
                clean_up_tokenization_spaces=True
            )
            return summary_result[0]['summary_text']
        except Exception as e:
            logging.error(f"Summarization failed: {str(e)}")
            sentences = text.split('. ')
            return sentences[0] + "." if sentences else "Summary not available."

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
