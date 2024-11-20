import os
import logging
from pydub import AudioSegment
from whisper_api import WhisperAPI
from roberta_api import HuggingFaceAPI
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Tuple, List, Dict
import time
import backoff
import asyncio
import tempfile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, whisper_api_key: str, huggingface_api_key: str):
        self.whisper = WhisperAPI(whisper_api_key)
        self.predictor = HuggingFaceAPI(huggingface_api_key)
        self.chunk_length_ms = 20 * 1000  # 20 seconds
        self.overlap_ms = 2000  # 2 second overlap
        self.min_chunk_length_ms = 5000  # 5 seconds minimum
        self.max_workers = 8
        self.temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory: {self.temp_dir}")

    def __del__(self):
        """Cleanup temporary directory on object destruction"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.debug(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {str(e)}")

    def get_temp_path(self, suffix: str) -> str:
        """Generate temp file path"""
        return os.path.join(self.temp_dir, f"{int(time.time())}_{suffix}")

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def process_chunk_async(self, chunk: AudioSegment, chunk_index: int) -> Tuple[str, str, float]:
        """Process a single chunk asynchronously"""
        chunk_path = self.get_temp_path(f"chunk_{chunk_index}.wav")
        try:
            # Export chunk with optimized settings
            chunk.export(
                chunk_path, 
                format="wav",
                parameters=["-ac", "1", "-ar", "16000"]
            )
            
            # Process concurrently
            transcription = await self.whisper.transcribe_async(chunk_path)
            prediction, confidence = await self.predictor.predict_async(transcription)
            
            return transcription, prediction, confidence
            
        finally:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    def optimize_audio(self, audio: AudioSegment) -> AudioSegment:
        """Optimize audio settings"""
        return audio.set_channels(1).set_frame_rate(16000)

    def get_weighted_prediction(self, chunk_results: List[Dict]) -> Tuple[str, float]:
        """Get final prediction using weighted voting"""
        if not chunk_results:
            return "Safe", 0.0

        predictions = np.array([
            (1 if r['prediction'] == "Phish" else 0, r['confidence']) 
            for r in chunk_results
        ])
        
        phish_score = np.sum(predictions[predictions[:, 0] == 1, 1])
        safe_score = np.sum(predictions[predictions[:, 0] == 0, 1])
        
        total_score = phish_score + safe_score
        if total_score == 0:
            return "Safe", 0.0
            
        if phish_score > safe_score:
            return "Phish", phish_score / total_score
        else:
            return "Safe", safe_score / total_score

    async def process_audio_async(self, audio_path: str) -> Tuple[str, str, float]:
        """Process audio file asynchronously"""
        try:
            # Load and optimize audio
            audio = AudioSegment.from_file(audio_path)
            audio = self.optimize_audio(audio)
            duration_ms = len(audio)
            
            # Handle short files
            if duration_ms <= self.chunk_length_ms:
                temp_path = self.get_temp_path("short.wav")
                try:
                    audio.export(temp_path, format="wav", 
                               parameters=["-ac", "1", "-ar", "16000"])
                    transcription = await self.whisper.transcribe_async(temp_path)
                    prediction, confidence = await self.predictor.predict_async(transcription)
                    return transcription.strip(), prediction, confidence
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Process longer files in chunks
            chunks = []
            for i in range(0, duration_ms, self.chunk_length_ms - self.overlap_ms):
                chunk = audio[i:i + self.chunk_length_ms]
                if len(chunk) >= self.min_chunk_length_ms:
                    chunks.append((chunk, i))
            
            # Process chunks concurrently
            tasks = []
            async with asyncio.TaskGroup() as group:
                for i, (chunk, _) in enumerate(chunks):
                    task = group.create_task(self.process_chunk_async(chunk, i))
                    tasks.append((i, task))
            
            # Collect results
            chunk_results = []
            for i, task in tasks:
                try:
                    transcription, prediction, confidence = await task
                    chunk_results.append({
                        'transcription': transcription,
                        'prediction': prediction,
                        'confidence': confidence,
                        'chunk_index': i
                    })
                except Exception as e:
                    logger.error(f"Chunk {i} failed: {str(e)}")
            
            if not chunk_results:
                raise Exception("All chunks failed to process")
            
            # Sort and combine results
            chunk_results.sort(key=lambda x: x['chunk_index'])
            full_transcription = " ".join(r['transcription'] for r in chunk_results)
            
            # Get final prediction
            final_prediction, confidence = self.get_weighted_prediction(chunk_results)
            
            return full_transcription.strip(), final_prediction, confidence
            
        except Exception as e:
            logger.error(f"Error in process_audio: {str(e)}")
            raise

    def process_audio(self, audio_path: str) -> Tuple[str, str, float]:
        """Synchronous wrapper for async processing"""
        return asyncio.run(self.process_audio_async(audio_path))