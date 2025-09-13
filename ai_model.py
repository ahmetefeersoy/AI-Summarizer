import asyncio
import logging
from typing import Optional
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from dotenv import load_dotenv

load_dotenv()

class LocalAIModel:
    def __init__(self):
        self.summarizer = None
        self.model_name = "facebook/bart-large-cnn"  
        self.max_length = 512
        self.min_length = 30
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
    async def initialize(self):
        """Initialize the local AI model"""
        try:
            logging.info(f"Loading AI model: {self.model_name} on {self.device}")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if self.device == "cuda" else -1,
                framework="pt"
            )
            
            self.model_loaded = True
            logging.info("AI model loaded successfully!")
            
        except Exception as e:
            logging.error(f"Failed to load AI model: {str(e)}")
            # Fallback to a lighter model if BART fails
            try:
                logging.info("Falling back to distilbart model...")
                self.model_name = "sshleifer/distilbart-cnn-12-6"
                self.summarizer = pipeline(
                    "summarization",
                    model=self.model_name,
                    device=0 if self.device == "cuda" else -1,
                    framework="pt"
                )
                self.model_loaded = True
                logging.info("Fallback model loaded successfully!")
            except Exception as fallback_error:
                logging.error(f"Fallback model also failed: {str(fallback_error)}")
                self.model_loaded = False
    
    async def summarize_text(self, text: str) -> str:
        """Summarize text using the local AI model"""
        if not self.model_loaded:
            return self._rule_based_summarize(text)
        
        try:
            text = text.strip()
            if len(text) < 50:
                return f"Summary: {text}"
            
            if len(text) > 1024:
                text = text[:1024] + "..."
            
            # Generate summary using the local model
            summary_result = self.summarizer(
                text,
                max_length=self.max_length,
                min_length=self.min_length,
                do_sample=False,
                truncation=True
            )
            
            summary = summary_result[0]['summary_text']
            return f"{summary}"
            
        except Exception as e:
            logging.error(f"AI summarization failed: {str(e)}")
            # Fallback to rule-based
            return self._rule_based_summarize(text)
    
    def _rule_based_summarize(self, text: str) -> str:
        """Fallback rule-based summarization"""
        sentences = text.split('. ')
        if len(sentences) <= 2:
            return f"Summary: {text}"
        
        # Return first and last sentences
        summary = f"{sentences[0]}. {sentences[-1]}"
        return f"Summary: {summary}"
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "max_length": self.max_length,
            "min_length": self.min_length
        }

# Global AI model instance
ai_model = LocalAIModel()
