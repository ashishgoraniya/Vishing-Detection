import requests
import logging
import backoff
from typing import Optional
import aiohttp
import os
import asyncio
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class WhisperAPI:
    def __init__(self, api_key: str):
        """Initialize WhisperAPI with OpenAI API key"""
        self.api_key = api_key
        self.api_url = "https://api.openai.com/v1/audio/transcriptions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.max_retries = 3
        self.retry_delay = 5
        self.timeout = 60  # 60 seconds timeout

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.HTTPError),
        max_tries=3,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and 
                        e.response.status_code in (400, 401, 403)
    )
    def transcribe(self, audio_file_path: str) -> str:
        """
        Synchronously transcribe audio file with retry logic
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            str: Transcribed text
            
        Raises:
            Exception: If transcription fails after retries
        """
        try:
            logger.debug(f"Starting transcription for: {audio_file_path}")
            
            with open(audio_file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {
                    'model': 'whisper-1',
                    'response_format': 'json',
                    'language': 'en'  # Specify English for better accuracy
                }
                
                for attempt in range(self.max_retries):
                    try:
                        response = requests.post(
                            self.api_url,
                            headers=self.headers,
                            files=files,
                            data=data,
                            timeout=self.timeout
                        )
                        
                        if response.status_code == 503:
                            logger.warning("Service temporarily unavailable, retrying...")
                            time.sleep(self.retry_delay)
                            continue
                            
                        response.raise_for_status()
                        result = response.json()
                        
                        if 'text' not in result:
                            raise ValueError("No transcription in response")
                            
                        logger.debug("Transcription successful")
                        return result['text']
                        
                    except requests.exceptions.RequestException as e:
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                            time.sleep(self.retry_delay)
                        else:
                            raise

        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

    async def transcribe_async(self, audio_file_path: str) -> str:
        """
        Asynchronously transcribe audio file
        
        Args:
            audio_file_path: Path to the audio file
            
        Returns:
            str: Transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        try:
            logger.debug(f"Starting async transcription for: {audio_file_path}")
            
            async with aiohttp.ClientSession() as session:
                with open(audio_file_path, 'rb') as f:
                    data = aiohttp.FormData()
                    data.add_field('file', f)
                    data.add_field('model', 'whisper-1')
                    data.add_field('response_format', 'json')
                    data.add_field('language', 'en')
                    
                    for attempt in range(self.max_retries):
                        try:
                            async with session.post(
                                self.api_url,
                                headers=self.headers,
                                data=data,
                                timeout=aiohttp.ClientTimeout(total=self.timeout)
                            ) as response:
                                if response.status == 503:
                                    logger.warning("Service temporarily unavailable, retrying...")
                                    await asyncio.sleep(self.retry_delay)
                                    continue
                                    
                                response.raise_for_status()
                                result = await response.json()
                                
                                if 'text' not in result:
                                    raise ValueError("No transcription in response")
                                    
                                logger.debug("Async transcription successful")
                                return result['text']
                                
                        except Exception as e:
                            if attempt < self.max_retries - 1:
                                logger.warning(f"Async attempt {attempt + 1} failed: {str(e)}")
                                await asyncio.sleep(self.retry_delay)
                            else:
                                raise

        except Exception as e:
            logger.error(f"Async transcription error: {str(e)}")
            raise

    def transcribe_long_audio(self, audio_file_path: str, chunk_duration: int = 30) -> str:
        """
        Transcribe long audio files by splitting into chunks
        
        Args:
            audio_file_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            str: Combined transcription
        """
        try:
            from pydub import AudioSegment
            
            logger.info(f"Processing long audio file: {audio_file_path}")
            
            # Load audio file
            audio = AudioSegment.from_file(audio_file_path)
            duration_seconds = len(audio) / 1000
            
            if duration_seconds <= chunk_duration:
                return self.transcribe(audio_file_path)
            
            # Process in chunks
            transcriptions = []
            chunk_length_ms = chunk_duration * 1000
            
            for i in range(0, len(audio), chunk_length_ms):
                chunk = audio[i:i + chunk_length_ms]
                chunk_path = f"temp_chunk_{i}.wav"
                
                try:
                    # Export chunk with optimized settings
                    chunk.export(
                        chunk_path,
                        format="wav",
                        parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz
                    )
                    
                    # Add small delay between chunks
                    time.sleep(1)
                    
                    # Transcribe chunk
                    chunk_transcription = self.transcribe(chunk_path)
                    transcriptions.append(chunk_transcription)
                    
                finally:
                    if os.path.exists(chunk_path):
                        os.remove(chunk_path)
            
            logger.info("Long audio processing complete")
            return " ".join(transcriptions)
            
        except Exception as e:
            logger.error(f"Error transcribing long audio: {str(e)}")
            raise

    async def transcribe_long_audio_async(self, audio_file_path: str, chunk_duration: int = 30) -> str:
        """
        Asynchronously transcribe long audio files
        
        Args:
            audio_file_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            str: Combined transcription
        """
        try:
            from pydub import AudioSegment
            
            logger.info(f"Processing long audio file asynchronously: {audio_file_path}")
            
            # Load and optimize audio
            audio = AudioSegment.from_file(audio_file_path)
            audio = audio.set_channels(1).set_frame_rate(16000)  # Optimize format
            
            if len(audio) <= chunk_duration * 1000:
                return await self.transcribe_async(audio_file_path)
            
            # Process chunks concurrently
            chunk_length_ms = chunk_duration * 1000
            tasks = []
            temp_files = []
            
            async with asyncio.TaskGroup() as group:
                for i in range(0, len(audio), chunk_length_ms):
                    chunk = audio[i:i + chunk_length_ms]
                    chunk_path = f"temp_chunk_{i}.wav"
                    temp_files.append(chunk_path)
                    
                    # Export chunk
                    chunk.export(
                        chunk_path,
                        format="wav",
                        parameters=["-ac", "1", "-ar", "16000"]
                    )
                    
                    # Create transcription task
                    task = group.create_task(self.transcribe_async(chunk_path))
                    tasks.append((i, task))
            
            # Collect results in order
            results = []
            for i, task in sorted(tasks, key=lambda x: x[0]):
                try:
                    transcription = await task
                    results.append(transcription)
                except Exception as e:
                    logger.error(f"Chunk {i} failed: {str(e)}")
            
            # Cleanup temp files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
            logger.info("Async long audio processing complete")
            return " ".join(results)
            
        except Exception as e:
            logger.error(f"Error in async long audio transcription: {str(e)}")
            raise