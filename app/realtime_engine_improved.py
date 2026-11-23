"""
Improved Real-Time Speaker Diarization Engine
Based on research: Online Speaker Diarization with Continuous Embedding Re-clustering

Key improvements:
1. Accumulate embeddings over time for better speaker profiles
2. Periodic re-clustering of all segments to fix early mistakes
3. Minimum segment length enforcement (3+ seconds preferred)
4. Speaker profile averaging with weighted updates
5. Confidence-based merging after session ends
"""
import torch
import numpy as np
import asyncio
from collections import deque
from typing import Optional, Dict, List, Tuple
import soundfile as sf
from pathlib import Path
import tempfile
import logging
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cosine

from app.vad_service import VADService
from app.asr_service import ASRService
from app.spk_service import SpeakerService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeakerProfile:
    """Tracks accumulated embeddings and statistics for a speaker"""
    def __init__(self, speaker_id: str, initial_embedding: np.ndarray):
        self.speaker_id = speaker_id
        self.embeddings = [initial_embedding]  # Store all embeddings
        self.mean_embedding = initial_embedding.copy()
        self.segment_count = 1
        self.total_duration = 0.0
        
    def add_embedding(self, embedding: np.ndarray, duration: float = 0.0):
        """Add new embedding with incremental averaging"""
        self.embeddings.append(embedding)
        self.segment_count += 1
        self.total_duration += duration
        
        # Weighted average: recent embeddings get slightly more weight
        alpha = 0.3  # Weight for new embedding
        self.mean_embedding = (1 - alpha) * self.mean_embedding + alpha * embedding
        
    def similarity_to(self, embedding: np.ndarray) -> float:
        """Calculate cosine similarity to this speaker"""
        return 1.0 - cosine(self.mean_embedding, embedding)
        
    def merge_with(self, other: 'SpeakerProfile'):
        """Merge another speaker profile into this one"""
        self.embeddings.extend(other.embeddings)
        self.segment_count += other.segment_count
        self.total_duration += other.total_duration
        
        # Recalculate mean from all embeddings
        all_embeddings = np.array(self.embeddings)
        self.mean_embedding = np.mean(all_embeddings, axis=0)


class ImprovedRealtimeEngine:
    """
    Research-based real-time diarization with continuous re-clustering.
    
    Strategy:
    - Process segments incrementally as they arrive
    - Maintain speaker profiles with accumulated embeddings
    - Periodically re-cluster all segments (every 10 segments or 30s)
    - Final merge pass at end of session to consolidate similar speakers
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        vad_threshold: float = 0.4,
        min_speech_duration: float = 2.0,  # Longer minimum for better embeddings
        min_silence_duration: float = 0.8,
        similarity_threshold: float = 0.70,  # Higher threshold for initial match
        merge_threshold: float = 0.80,  # For final consolidation
        recluster_interval: int = 10,  # Re-cluster every N segments
        max_speakers: int = 2  # Maximum expected speakers (default 2 for conversations)
    ):
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.min_silence_duration = min_silence_duration
        self.similarity_threshold = similarity_threshold
        self.merge_threshold = merge_threshold
        self.recluster_interval = recluster_interval
        self.max_speakers = max_speakers
        
        logger.info(f"Improved real-time engine initialized:")
        logger.info(f"  - Min speech: {min_speech_duration}s (longer = better embeddings)")
        logger.info(f"  - Initial match threshold: {similarity_threshold}")
        logger.info(f"  - Merge threshold: {merge_threshold}")
        logger.info(f"  - Re-cluster every {recluster_interval} segments")
        logger.info(f"  - Max speakers: {max_speakers}")
        
        # Initialize services
        logger.info("Initializing VAD service...")
        self.vad_service = VADService(sample_rate=sample_rate)
        
        logger.info("Initializing ASR service...")
        self.asr_service = ASRService()
        
        logger.info("Initializing Speaker Recognition service...")
        self.speaker_service = SpeakerService()
        
        # Audio buffer for sliding window
        self.audio_buffer = deque(maxlen=int(sample_rate * 60))  # 60s buffer
        self.buffer_timestamps = deque(maxlen=int(sample_rate * 60))
        self.current_time = 0.0
        
        # VAD state
        self.speech_start_time = None
        self.silence_duration = 0.0
        
        # Speaker profiles and segments
        self.speaker_profiles: Dict[str, SpeakerProfile] = {}
        self.processed_segments = []  # List of (start, end, speaker_id, text, embedding)
        self.next_speaker_id = 1
        self.segments_since_recluster = 0
        
    async def process_audio_chunk(self, audio_chunk: np.ndarray):
        """Process incoming audio chunk with VAD"""
        # Add to buffer
        chunk_duration = len(audio_chunk) / self.sample_rate
        for i, sample in enumerate(audio_chunk):
            self.audio_buffer.append(sample)
            self.buffer_timestamps.append(self.current_time + i / self.sample_rate)
        self.current_time += chunk_duration
        
        # Run VAD
        vad_probs = self.vad_service.process_chunk(audio_chunk)
        is_speech = vad_probs > self.vad_threshold
        
        if is_speech:
            if self.speech_start_time is None:
                self.speech_start_time = self.current_time - chunk_duration
                logger.info(f"Speech started at {self.speech_start_time:.2f}s")
            self.silence_duration = 0.0
        else:
            if self.speech_start_time is not None:
                self.silence_duration += chunk_duration
                if self.silence_duration == chunk_duration:  # First silence chunk
                    logger.info(f"Silence detected after speech, waiting {self.min_silence_duration}s...")
                
                if self.silence_duration >= self.min_silence_duration:
                    speech_end = self.current_time - self.silence_duration
                    duration = speech_end - self.speech_start_time
                    
                    if duration >= self.min_speech_duration:
                        logger.info(f"✓ Speech segment detected: {self.speech_start_time:.2f}s - {speech_end:.2f}s ({duration:.2f}s)")
                        segment = await self._process_segment(self.speech_start_time, speech_end)
                        
                        # Check if we should re-cluster
                        self.segments_since_recluster += 1
                        if self.segments_since_recluster >= self.recluster_interval:
                            logger.info("Re-clustering all segments for improved accuracy...")
                            self._recluster_all_segments()
                            self.segments_since_recluster = 0
                        
                        self.speech_start_time = None
                        self.silence_duration = 0.0
                        return segment
                    else:
                        logger.warning(f"✗ Segment too short: {duration:.2f}s < {self.min_speech_duration}s (DISCARDED)")
                        self.speech_start_time = None
                        self.silence_duration = 0.0
        
        return None
        
    async def _process_segment(self, start_time: float, end_time: float):
        """Process a complete speech segment"""
        # Extract audio
        audio = self._extract_audio_segment(start_time, end_time)
        if len(audio) < int(self.sample_rate * 0.5):
            return None
            
        # Normalize and pad
        audio = audio.astype(np.float32)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        min_samples = int(self.sample_rate * 2.0)
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)), mode='constant')
            
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            sf.write(tmp_path, audio, self.sample_rate)
            
        try:
            # Transcribe
            text = self.asr_service.transcribe_file(tmp_path)
            
            # Get embedding
            embedding = self.speaker_service.get_embedding_file(tmp_path)
            
            # Match or create speaker
            speaker_id = self._match_or_create_speaker(embedding, end_time - start_time)
            
            # Store segment
            self.processed_segments.append({
                'start': start_time,
                'end': end_time,
                'speaker': speaker_id,
                'text': text,
                'embedding': embedding
            })
            
            logger.info(f"[{speaker_id}] {text}")
            
            return {
                'start': start_time,
                'end': end_time,
                'speaker': speaker_id,
                'speaker_id': speaker_id,
                'text': text,
                'confidence': 0.75,
                'speaker_confidence': 0.75
            }
            
        finally:
            try:
                Path(tmp_path).unlink()
            except:
                pass
                
    def _extract_audio_segment(self, start_time: float, end_time: float) -> np.ndarray:
        """Extract audio between timestamps from buffer"""
        buffer_array = np.array(self.audio_buffer)
        timestamps_array = np.array(self.buffer_timestamps)
        
        mask = (timestamps_array >= start_time) & (timestamps_array <= end_time)
        return buffer_array[mask]
        
    def _match_or_create_speaker(self, embedding: np.ndarray, duration: float) -> str:
        """Match to existing speaker or create new one"""
        if not self.speaker_profiles:
            # First speaker
            speaker_id = f"Speaker_{self.next_speaker_id}"
            self.next_speaker_id += 1
            self.speaker_profiles[speaker_id] = SpeakerProfile(speaker_id, embedding)
            logger.info(f"★ FIRST SPEAKER: {speaker_id}")
            return speaker_id
            
        # Find best match
        best_speaker = None
        best_similarity = -1.0
        
        for sid, profile in self.speaker_profiles.items():
            sim = profile.similarity_to(embedding)
            logger.debug(f"  Similarity to {sid}: {sim:.3f}")
            if sim > best_similarity:
                best_similarity = sim
                best_speaker = sid
                
        if best_similarity > self.similarity_threshold:
            # Match found
            logger.info(f"✓ MATCHED {best_speaker} (sim: {best_similarity:.3f})")
            self.speaker_profiles[best_speaker].add_embedding(embedding, duration)
            return best_speaker
        else:
            # New speaker
            speaker_id = f"Speaker_{self.next_speaker_id}"
            self.next_speaker_id += 1
            self.speaker_profiles[speaker_id] = SpeakerProfile(speaker_id, embedding)
            logger.info(f"✗ NEW SPEAKER: {speaker_id} (best_sim: {best_similarity:.3f})")
            return speaker_id
            
    def _recluster_all_segments(self):
        """Re-cluster all processed segments using accumulated embeddings"""
        if len(self.processed_segments) < 3:
            return
            
        logger.info(f"Re-clustering {len(self.processed_segments)} segments...")
        
        # Extract embeddings
        embeddings = np.array([seg['embedding'] for seg in self.processed_segments])
        
        # Use max_speakers if specified, otherwise auto-detect
        if self.max_speakers:
            best_n_speakers = min(self.max_speakers, len(self.processed_segments))
            logger.info(f"  Using fixed {best_n_speakers} speakers (max_speakers={self.max_speakers})")
        else:
            # Try different numbers of speakers and pick best
            best_n_speakers = 2
            best_score = -1.0
            
            # Prefer 2 speakers for conversations
            for n in range(2, min(5, len(self.processed_segments))):
                clustering = AgglomerativeClustering(
                    n_clusters=n,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(embeddings)
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                try:
                    score = silhouette_score(embeddings, labels, metric='cosine')
                    logger.info(f"  n={n} speakers: score={score:.3f}")
                    
                    # Bias towards 2 speakers (add bonus for n=2)
                    adjusted_score = score + (0.1 if n == 2 else 0.0)
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_n_speakers = n
                except:
                    pass
                
        # Apply clustering with determined number of speakers
        clustering = AgglomerativeClustering(
            n_clusters=best_n_speakers,
            metric='cosine',
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        logger.info(f"✓ Re-clustered into {best_n_speakers} speakers")
        
        # Rebuild speaker profiles
        self.speaker_profiles = {}
        speaker_map = {}
        
        for cluster_id in range(best_n_speakers):
            speaker_id = f"Speaker_{cluster_id + 1}"
            cluster_embeddings = embeddings[labels == cluster_id]
            mean_embedding = np.mean(cluster_embeddings, axis=0)
            
            profile = SpeakerProfile(speaker_id, mean_embedding)
            profile.embeddings = list(cluster_embeddings)
            profile.segment_count = len(cluster_embeddings)
            
            self.speaker_profiles[speaker_id] = profile
            speaker_map[cluster_id] = speaker_id
            
        # Update segment labels
        for i, seg in enumerate(self.processed_segments):
            seg['speaker'] = speaker_map[labels[i]]
            seg['speaker_id'] = speaker_map[labels[i]]
            
        self.next_speaker_id = best_n_speakers + 1
        
    def finalize_session(self):
        """Final merge pass to consolidate very similar speakers"""
        if len(self.speaker_profiles) <= 1:
            return
            
        logger.info("Finalizing session: merging similar speakers...")
        
        # If we have many speakers, do aggressive hierarchical merging
        if len(self.speaker_profiles) > 3:
            logger.info(f"Too many speakers ({len(self.speaker_profiles)}), forcing 2-speaker clustering...")
            
            # Extract all embeddings
            all_embeddings = []
            segment_indices = []
            for i, seg in enumerate(self.processed_segments):
                all_embeddings.append(seg['embedding'])
                segment_indices.append(i)
            
            if len(all_embeddings) >= 2:
                # Force 2-cluster solution
                clustering = AgglomerativeClustering(
                    n_clusters=2,
                    metric='cosine',
                    linkage='average'
                )
                labels = clustering.fit_predict(np.array(all_embeddings))
                
                # Rebuild speaker profiles
                self.speaker_profiles = {}
                for cluster_id in range(2):
                    speaker_id = f"Speaker_{cluster_id + 1}"
                    cluster_embeddings = [all_embeddings[i] for i in range(len(labels)) if labels[i] == cluster_id]
                    mean_embedding = np.mean(cluster_embeddings, axis=0)
                    
                    profile = SpeakerProfile(speaker_id, mean_embedding)
                    profile.embeddings = cluster_embeddings
                    profile.segment_count = len(cluster_embeddings)
                    self.speaker_profiles[speaker_id] = profile
                
                # Update all segments
                for i, seg in enumerate(self.processed_segments):
                    new_speaker = f"Speaker_{labels[i] + 1}"
                    seg['speaker'] = new_speaker
                    seg['speaker_id'] = new_speaker
                
                logger.info(f"Forced re-clustering to 2 speakers")
                return
        
        # Calculate pairwise similarities
        speaker_ids = list(self.speaker_profiles.keys())
        similarities = np.zeros((len(speaker_ids), len(speaker_ids)))
        
        for i, sid1 in enumerate(speaker_ids):
            for j, sid2 in enumerate(speaker_ids):
                if i < j:
                    sim = self.speaker_profiles[sid1].similarity_to(
                        self.speaker_profiles[sid2].mean_embedding
                    )
                    similarities[i, j] = sim
                    similarities[j, i] = sim
                    logger.info(f"Similarity {sid1} <-> {sid2}: {sim:.3f}")
                    
        # Merge speakers above merge threshold
        merged = set()
        merge_map = {}
        
        for i, sid1 in enumerate(speaker_ids):
            if sid1 in merged:
                continue
            merge_map[sid1] = sid1
            
            for j, sid2 in enumerate(speaker_ids):
                if i < j and sid2 not in merged:
                    if similarities[i, j] > self.merge_threshold:
                        logger.info(f"Merging {sid2} into {sid1} (sim: {similarities[i, j]:.3f})")
                        self.speaker_profiles[sid1].merge_with(self.speaker_profiles[sid2])
                        merged.add(sid2)
                        merge_map[sid2] = sid1
                        
        # Remove merged speakers
        for sid in merged:
            del self.speaker_profiles[sid]
            
        # Update segment labels
        for seg in self.processed_segments:
            if seg['speaker'] in merge_map:
                new_speaker = merge_map[seg['speaker']]
                seg['speaker'] = new_speaker
                seg['speaker_id'] = new_speaker
                
        logger.info(f"Final speaker count: {len(self.speaker_profiles)}")
        
    def get_conversation_history(self):
        """Get all processed segments"""
        return [
            {
                'start': seg['start'],
                'end': seg['end'],
                'speaker': seg['speaker'],
                'speaker_id': seg['speaker_id'],
                'text': seg['text'],
                'confidence': 0.75,
                'speaker_confidence': 0.75
            }
            for seg in self.processed_segments
        ]
        
    def reset(self):
        """Reset engine state"""
        self.speaker_profiles.clear()
        self.processed_segments.clear()
        self.speech_start_time = None
        self.silence_duration = 0.0
        self.next_speaker_id = 1
        self.segments_since_recluster = 0
        logger.info("Engine reset")
