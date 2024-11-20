import requests
import logging
import backoff
from typing import Tuple, List
import time
import numpy as np
import aiohttp

logger = logging.getLogger(__name__)

class HuggingFaceAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api-inference.huggingface.co/models/itxag17/Audio-Phishing-Detection"
        # self.api_url = "https://api-inference.huggingface.co/models/itxag17/roberta"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.max_text_length = 512
        self.max_retries = 3
        self.retry_delay = 5

    def preprocess_text(self, text: str) -> str:
        """Clean and prepare text for prediction"""
        import re
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove repeated phrases
        text = re.sub(r'(\b\w+\b)(\s+\1\b)+', r'\1', text)
        
        # Clean up punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text

    def split_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word.split()) > self.max_text_length:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word.split())
            else:
                current_chunk.append(word)
                current_length += len(word.split())
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=3
    )
    def predict(self, text: str) -> Tuple[str, float]:
        """Synchronous prediction"""
        try:
            processed_text = self.preprocess_text(text)
            chunks = self.split_text(processed_text)
            predictions = []
            
            for chunk in chunks:
                payload = {"inputs": chunk}
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 503:
                    logger.warning("Model is loading, waiting...")
                    time.sleep(self.retry_delay)
                    continue
                    
                response.raise_for_status()
                result = response.json()
                prediction = result[0][0]
                label = prediction['label']
                confidence = prediction['score']
                
                predictions.append(("Phish" if label == "LABEL_1" else "Safe", confidence))
            
            if not predictions:
                return "Safe", 0.0
                
            # Combine predictions
            phish_score = sum(conf for pred, conf in predictions if pred == "Phish")
            safe_score = sum(conf for pred, conf in predictions if pred == "Safe")
            
            total_score = phish_score + safe_score
            if total_score == 0:
                return "Safe", 0.0
            
            if phish_score > safe_score:
                return "Phish", phish_score / total_score
            else:
                return "Safe", safe_score / total_score
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

    async def predict_async(self, text: str) -> Tuple[str, float]:
        """Asynchronous prediction"""
        try:
            processed_text = self.preprocess_text(text)
            chunks = self.split_text(processed_text)
            predictions = []
            
            async with aiohttp.ClientSession() as session:
                for chunk in chunks:
                    for attempt in range(self.max_retries):
                        try:
                            payload = {"inputs": chunk}
                            async with session.post(
                                self.api_url,
                                headers=self.headers,
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)
                            ) as response:
                                if response.status == 503:
                                    logger.warning("Model is loading, waiting...")
                                    await asyncio.sleep(self.retry_delay)
                                    continue
                                    
                                response.raise_for_status()
                                result = await response.json()
                                prediction = result[0][0]
                                label = prediction['label']
                                confidence = prediction['score']
                                
                                predictions.append(
                                    ("Phish" if label == "LABEL_1" else "Safe", 
                                     confidence)
                                )
                                break
                                
                        except Exception as e:
                            if attempt == self.max_retries - 1:
                                raise
                            await asyncio.sleep(self.retry_delay)
            
            if not predictions:
                return "Safe", 0.0
                
            # Combine predictions
            phish_score = sum(conf for pred, conf in predictions if pred == "Phish")
            safe_score = sum(conf for pred, conf in predictions if pred == "Safe")
            
            total_score = phish_score + safe_score
            if total_score == 0:
                return "Safe", 0.0
            
            if phish_score > safe_score:
                return "Phish", phish_score / total_score
            else:
                return "Safe", safe_score / total_score
                
        except Exception as e:
            logger.error(f"Async prediction error: {str(e)}")
            raise