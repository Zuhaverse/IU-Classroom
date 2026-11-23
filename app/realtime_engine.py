"""
Real-time Voice Intelligence Engine for Dental Clinics
Handles streaming audio from dual microphones with live VAD, diarization, ASR, and speaker identification.
"""
import torch
import numpy as np
import asyncio
import queue
from collections import deque
from typing import Optional, Dict, List, Tuple
import soundfile as sf
from pathlib import Path
import tempfile
import logging

from app.vad_service import VADService
from app.asr_service import ASRService
from app.spk_service import SpeakerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealtimeAudioBuffer:
    """Manages incoming audio chunks and maintains sliding window"""
    
    def __init__(self, sample_rate: int = 16000, max_duration: float = 30.0):
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = deque(maxlen=self.max_samples)
        self.timestamps = deque(maxlen=self.max_samples)
        self.current_time = 0.0
        
    def add_chunk(self, audio_chunk: np.ndarray):
        """Add new audio chunk to buffer"""
        chunk_duration = len(audio_chunk) / self.sample_rate
        for sample in audio_chunk:
            self.buffer.append(sample)
            self.timestamps.append(self.current_time)
            self.current_time += 1.0 / self.sample_rate
            
    def get_recent_audio(self, duration: float) -> Tuple[np.ndarray, float]:
        """Get recent audio of specified duration"""
        n_samples = int(duration * self.sample_rate)
        n_samples = min(n_samples, len(self.buffer))
        
        if n_samples == 0:
            return np.array([]), 0.0
            
        recent_audio = np.array(list(self.buffer)[-n_samples:])
        start_time = list(self.timestamps)[-n_samples] if n_samples > 0 else self.current_time
        
        return recent_audio, start_time
        
    def get_audio_segment(self, start_time: float, end_time: float) -> np.ndarray:
        """Extract audio segment by timestamp"""
        timestamps_list = list(self.timestamps)
        buffer_list = list(self.buffer)
        
        start_idx = None
        end_idx = None
        
        for i, ts in enumerate(timestamps_list):
            if start_idx is None and ts >= start_time:
                start_idx = i
            if ts <= end_time:
                end_idx = i
                
        if start_idx is None or end_idx is None:
            return np.array([])
            
        return np.array(buffer_list[start_idx:end_idx+1])


class RealtimeTranscriptSegment:
    """Represents a single transcribed segment with metadata"""
    
    def __init__(self, start_time: float, end_time: float, speaker_id: str, 
                 text: str, confidence: float, speaker_confidence: float):
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id
        self.text = text
        self.confidence = confidence
        self.speaker_confidence = speaker_confidence
        
    def to_dict(self) -> Dict:
        return {
            'start': float(self.start_time),
            'end': float(self.end_time),
            'speaker': self.speaker_id,
            'text': self.text,
            'confidence': float(self.confidence),
            'speaker_confidence': float(self.speaker_confidence)
        }


class RealtimeVoiceEngine:
    """
    Main real-time processing engine.
    Processes streaming audio with VAD, diarization, ASR, and speaker identification.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.5,
        min_speech_duration: float = 0.3,
        min_silence_duration: float = 0.5,
        lookback_duration: float = 30.0,
        speaker_similarity_threshold: float = 0.45  # Lowered for more aggressive separation
    ):
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.speaker_similarity_threshold = speaker_similarity_threshold
        logger.info(f"Speaker clustering threshold set to {self.speaker_similarity_threshold}")
        
        # Initialize services
        logger.info("Initializing VAD service...")
        self.vad_service = VADService(sample_rate=sample_rate)
        
        logger.info("Initializing ASR service...")
        self.asr_service = ASRService()
        
        logger.info("Initializing Speaker Recognition service...")
        self.speaker_service = SpeakerService()
        
        # Audio buffer
        self.audio_buffer = RealtimeAudioBuffer(
            sample_rate=sample_rate,
            max_duration=lookback_duration
        )
        
        # Speech activity tracking
        self.current_speech_segment = None
        self.speech_start_time = None
        self.silence_duration = 0.0
        self.min_silence_samples = int(min_silence_duration * sample_rate)
        
        # Results queue
        self.transcript_segments: List[RealtimeTranscriptSegment] = []
        self.pending_audio_segments: queue.Queue = queue.Queue()
        
        # Dynamic speaker clustering (for unknown speakers)
        self.speaker_embeddings = []  # List of (speaker_id, embedding)
        self.next_speaker_id = 0
        
        # State
        self.is_processing = False
        
    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[RealtimeTranscriptSegment]:
        """
        Process a single audio chunk from microphone input.
        Returns a transcript segment if speech segment is complete.
        
        Args:
            audio_chunk: Audio samples (mono, 16kHz)
            
        Returns:
            RealtimeTranscriptSegment if a complete utterance was detected, None otherwise
        """
        # Add to buffer
        self.audio_buffer.add_chunk(audio_chunk)
        
        # Run VAD on current chunk
        vad_probs = self.vad_service.process_chunk(audio_chunk)
        is_speech = vad_probs > self.vad_threshold
        
        current_time = self.audio_buffer.current_time
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        # Speech activity detection state machine
        if is_speech:
            if self.speech_start_time is None:
                # Speech started
                self.speech_start_time = current_time - chunk_duration
                logger.info(f"Speech started at {self.speech_start_time:.2f}s")
            
            # Reset silence counter
            self.silence_duration = 0.0
            
        else:  # silence
            if self.speech_start_time is not None:
                # We're in a potential speech segment
                self.silence_duration += chunk_duration
                
                # Check if silence duration exceeds threshold
                if self.silence_duration >= self.min_silence_duration:
                    speech_end_time = current_time - self.silence_duration
                    speech_duration = speech_end_time - self.speech_start_time
                    
                    # Check if speech duration is sufficient
                    if speech_duration >= self.min_speech_duration:
                        logger.info(f"Speech ended at {speech_end_time:.2f}s (duration: {speech_duration:.2f}s)")
                        
                        # Extract audio segment
                        segment_audio = self.audio_buffer.get_audio_segment(
                            self.speech_start_time,
                            speech_end_time
                        )
                        
                        # Process segment (transcribe + identify speaker)
                        if len(segment_audio) > 0:
                            result = await self._process_speech_segment(
                                segment_audio,
                                self.speech_start_time,
                                speech_end_time
                            )
                            
                            # Reset state
                            self.speech_start_time = None
                            self.silence_duration = 0.0
                            
                            return result
                    else:
                        # Too short, discard
                        logger.debug(f"Speech segment too short ({speech_duration:.2f}s), discarding")
                        self.speech_start_time = None
                        self.silence_duration = 0.0
        
        return None
        
    async def _process_speech_segment(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float
    ) -> RealtimeTranscriptSegment:
        """
        Process a complete speech segment: transcribe and identify speaker.
        Uses enrolled speakers if available, otherwise clusters unknown speakers.
        """
        # --- Normalize and pad audio segment ---
        import numpy as np
        segment_audio = audio.astype(np.float32)
        # Normalize to [-1, 1]
        max_val = np.abs(segment_audio).max()
        if max_val > 0:
            segment_audio = segment_audio / max_val
        # Pad to at least 2 seconds (32000 samples at 16kHz)
        min_samples = int(self.sample_rate * 2.0)
        if len(segment_audio) < min_samples:
            segment_audio = np.pad(segment_audio, (0, min_samples - len(segment_audio)), mode='constant')
        segment_duration = len(segment_audio) / self.sample_rate
        logger.info(f"Segment duration after normalization/padding: {segment_duration:.2f}s")

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, segment_audio, self.sample_rate)

        try:
            # Transcribe
            logger.info(f"Transcribing segment {start_time:.2f}s - {end_time:.2f}s")
            text = self.asr_service.transcribe_file(tmp_path)

            # Get speaker embedding
            logger.info(f"Extracting speaker embedding...")
            embedding = self.speaker_service.get_embedding_file(tmp_path)

            # First, try to match with enrolled speakers
            enrolled_id, enrolled_confidence = self.speaker_service.match_embedding(embedding)

            speaker_id = None
            speaker_confidence = 0.0

            # Use enrolled speaker if confidence is high
            if enrolled_id and enrolled_confidence > 0.75:
                speaker_id = enrolled_id
                speaker_confidence = enrolled_confidence
                logger.info(f"Matched enrolled speaker: {speaker_id} (confidence: {speaker_confidence:.3f})")
            else:
                # No enrolled match - use clustering for unknown speakers
                speaker_id, speaker_confidence = self._cluster_speaker(embedding)
                logger.info(f"Clustered as: {speaker_id} (confidence: {speaker_confidence:.3f})")

            # Create segment
            segment = RealtimeTranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id,
                text=text,
                confidence=1.0,  # ASR confidence can be added if model supports it
                speaker_confidence=speaker_confidence
            )

            self.transcript_segments.append(segment)
            logger.info(f"Segment complete: [{segment.speaker_id}] {segment.text}")

            return segment

        finally:
            # Cleanup
            try:
                Path(tmp_path).unlink()
            except:
                pass
                
    def _cluster_speaker(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Cluster speaker based on embedding similarity.
        Higher threshold = better grouping for 2-person conversations.

        Returns:
            (speaker_id, confidence)
        """
        best_match_id = None
        best_similarity = -1.0
        all_similarities = []
        
        # Compare with existing speaker embeddings
        for spk_id, spk_emb in self.speaker_embeddings:
            similarity = np.dot(embedding, spk_emb) / (
                np.linalg.norm(embedding) * np.linalg.norm(spk_emb)
            )
            all_similarities.append((spk_id, similarity))
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = spk_id
        
        # Log all similarities for debugging
        if all_similarities:
            sim_str = ', '.join([f"{sid}={sim:.3f}" for sid, sim in all_similarities])
            logger.info(f"Speaker comparison: [{sim_str}]")
        
        # If similarity is high enough, reuse existing speaker ID
        if best_similarity > self.speaker_similarity_threshold:
            logger.info(f"✓ MATCHED {best_match_id} (similarity: {best_similarity:.3f} > threshold: {self.speaker_similarity_threshold})")
            # Update the embedding with a running average to improve future matches
            for i, (spk_id, spk_emb) in enumerate(self.speaker_embeddings):
                if spk_id == best_match_id:
                    # Running average: 80% old, 20% new
                    self.speaker_embeddings[i] = (spk_id, 0.8 * spk_emb + 0.2 * embedding)
                    break
            return best_match_id, best_similarity
        
        # Otherwise, create new speaker
        new_speaker_id = f"Speaker_{self.next_speaker_id + 1}"
        self.next_speaker_id += 1
        self.speaker_embeddings.append((new_speaker_id, embedding))
        
        if len(self.speaker_embeddings) > 1:
            logger.info(f"✗ NEW SPEAKER: {new_speaker_id} (best_similarity: {best_similarity:.3f} < threshold: {self.speaker_similarity_threshold})")
            if best_similarity > 0.60:
                logger.warning(f"⚠ Close match! Consider increasing threshold if too many speakers detected")
        else:
            logger.info(f"★ FIRST SPEAKER: {new_speaker_id}")
        
        return new_speaker_id, 0.5
                
    def get_conversation_history(self, duration: Optional[float] = None) -> List[Dict]:
        """
        Get conversation history.
        
        Args:
            duration: If specified, only return segments from last N seconds
            
        Returns:
            List of segment dictionaries
        """
        if duration is None:
            return [seg.to_dict() for seg in self.transcript_segments]
        
        current_time = self.audio_buffer.current_time
        cutoff_time = current_time - duration
        
        return [
            seg.to_dict() 
            for seg in self.transcript_segments 
            if seg.start_time >= cutoff_time
        ]
        
    def reset(self):
        """Reset engine state"""
        self.transcript_segments.clear()
        self.speech_start_time = None
        self.silence_duration = 0.0
        self.speaker_embeddings.clear()
        self.next_speaker_id = 0
        logger.info("Engine reset")
        
    def get_speaker_stats(self) -> Dict:
        """Get statistics about detected speakers"""
        speaker_counts = {}
        for seg in self.transcript_segments:
            speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
        
        return {
            'total_speakers': len(self.speaker_embeddings),
            'speaker_segments': speaker_counts,
            'enrolled_speakers': len(self.speaker_service.enrollments)
        }


class DualMicrophoneEngine(RealtimeVoiceEngine):
    """
    Extended engine for dual microphone setup (dentist + patient).
    Handles stereo input or two separate streams.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Separate buffers for each channel
        self.buffer_left = RealtimeAudioBuffer(
            sample_rate=self.sample_rate,
            max_duration=30.0
        )
        self.buffer_right = RealtimeAudioBuffer(
            sample_rate=self.sample_rate,
            max_duration=30.0
        )
        
        # Separate VAD state for each channel
        self.speech_start_left = None
        self.speech_start_right = None
        self.silence_duration_left = 0.0
        self.silence_duration_right = 0.0
        
    async def process_stereo_chunk(
        self,
        left_channel: np.ndarray,
        right_channel: np.ndarray
    ) -> List[RealtimeTranscriptSegment]:
        """
        Process stereo audio chunk (left=dentist, right=patient).
        
        Returns:
            List of completed transcript segments (0-2 segments)
        """
        results = []
        
        # Process left channel (dentist)
        result_left = await self._process_channel(
            left_channel,
            self.buffer_left,
            'speech_start_left',
            'silence_duration_left',
            channel_id='Dentist'
        )
        if result_left:
            results.append(result_left)
            
        # Process right channel (patient)
        result_right = await self._process_channel(
            right_channel,
            self.buffer_right,
            'speech_start_right',
            'silence_duration_right',
            channel_id='Patient'
        )
        if result_right:
            results.append(result_right)
            
        return results
        
    async def _process_channel(
        self,
        audio_chunk: np.ndarray,
        buffer: RealtimeAudioBuffer,
        start_attr: str,
        silence_attr: str,
        channel_id: str
    ) -> Optional[RealtimeTranscriptSegment]:
        """Process a single channel with separate VAD state"""
        buffer.add_chunk(audio_chunk)
        
        vad_probs = self.vad_service.process_chunk(audio_chunk)
        is_speech = vad_probs > self.vad_threshold
        
        current_time = buffer.current_time
        chunk_duration = len(audio_chunk) / self.sample_rate
        
        speech_start = getattr(self, start_attr)
        silence_duration = getattr(self, silence_attr)
        
        if is_speech:
            if speech_start is None:
                setattr(self, start_attr, current_time - chunk_duration)
                logger.info(f"[{channel_id}] Speech started at {current_time - chunk_duration:.2f}s")
            setattr(self, silence_attr, 0.0)
        else:
            if speech_start is not None:
                silence_duration += chunk_duration
                setattr(self, silence_attr, silence_duration)
                
                if silence_duration >= self.min_silence_duration:
                    speech_end = current_time - silence_duration
                    speech_duration = speech_end - speech_start
                    
                    if speech_duration >= self.min_speech_duration:
                        logger.info(f"[{channel_id}] Speech ended at {speech_end:.2f}s")
                        
                        segment_audio = buffer.get_audio_segment(speech_start, speech_end)
                        
                        if len(segment_audio) > 0:
                            result = await self._process_speech_segment_with_channel(
                                segment_audio,
                                speech_start,
                                speech_end,
                                channel_id
                            )
                            
                            setattr(self, start_attr, None)
                            setattr(self, silence_attr, 0.0)
                            
                            return result
                    else:
                        setattr(self, start_attr, None)
                        setattr(self, silence_attr, 0.0)
        
        return None
        
    async def _process_speech_segment_with_channel(
        self,
        audio: np.ndarray,
        start_time: float,
        end_time: float,
        channel_id: str
    ) -> RealtimeTranscriptSegment:
        """Process segment with predefined channel ID"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, self.sample_rate)
        
        try:
            text = self.asr_service.transcribe_file(tmp_path)
            embedding = self.speaker_service.get_embedding_file(tmp_path)
            _, speaker_confidence = self.speaker_service.match_embedding(embedding)
            
            segment = RealtimeTranscriptSegment(
                start_time=start_time,
                end_time=end_time,
                speaker_id=channel_id,
                text=text,
                confidence=1.0,
                speaker_confidence=speaker_confidence
            )
            
            self.transcript_segments.append(segment)
            logger.info(f"[{channel_id}] {text}")
            
            return segment
            
        finally:
            try:
                Path(tmp_path).unlink()
            except:
                pass
