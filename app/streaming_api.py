"""
WebSocket-based real-time audio streaming endpoint.
Supports dual microphone input and real-time transcription.
"""
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import asyncio
import numpy as np
import json
import logging
import struct
from datetime import datetime
import os
import wave
from threading import Lock

from app.realtime_engine import RealtimeVoiceEngine
from app.realtime_engine_improved import ImprovedRealtimeEngine
try:
    from app.realtime_engine import DualMicrophoneEngine
except ImportError:
    DualMicrophoneEngine = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioStreamManager:
    """
    Manages WebSocket connections for real-time audio streaming.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.engines: Dict[str, RealtimeVoiceEngine] = {}
        self.polling_tasks: Dict[str, asyncio.Task] = {}
        self.recorders: Dict[str, "SessionWavRecorder"] = {}

    def _get_base_dir(self) -> str:
        # app/streaming_api.py -> project root
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _create_session_wav_path(self, session_id: str) -> str:
        base_dir = self._get_base_dir()
        recordings_dir = os.path.join(base_dir, "recordings")
        os.makedirs(recordings_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{session_id}_{ts}.wav"
        return os.path.join(recordings_dir, filename)
        
    async def connect(
        self,
        websocket: WebSocket,
        session_id: str,
        mode: str = "single"
    ):
        """
        Accept WebSocket connection and initialize processing engine.
        
        Args:
            websocket: FastAPI WebSocket connection
            session_id: Unique session identifier
            mode: 'single' for single mic, 'dual' for dual microphone setup
        """
        await websocket.accept()
        self.active_connections[session_id] = websocket
        
        # Initialize appropriate engine
        if mode == "dual" and DualMicrophoneEngine:
            engine = DualMicrophoneEngine(
                sample_rate=16000,
                vad_threshold=0.3,
                min_speech_duration=0.3,
                min_silence_duration=0.3
            )
        else:
            # Use improved engine with re-clustering and speaker profile accumulation
            engine = ImprovedRealtimeEngine(
                sample_rate=16000,
                vad_threshold=0.3,  # Lower = more sensitive to speech
                min_speech_duration=1.5,  # Faster response, still good quality
                min_silence_duration=0.6,  # Detect pauses faster
                similarity_threshold=0.45,  # LOWERED - easier to match existing speakers
                merge_threshold=0.65,  # LOWERED - merge similar speakers more aggressively
                recluster_interval=4,  # More frequent re-clustering
                max_speakers=2  # Force 2-speaker model
            )
            
        self.engines[session_id] = engine

        # Initialize per-session WAV recorder
        channels = 2 if mode == "dual" else 1
        wav_path = self._create_session_wav_path(session_id)
        try:
            recorder = SessionWavRecorder(wav_path, sample_rate=engine.sample_rate, channels=channels)
            self.recorders[session_id] = recorder
            recording_status = "started"
        except Exception as rec_err:
            logger.error(f"Failed to start WAV recorder for {session_id}: {rec_err}")
            recording_status = "failed"
            wav_path = None
        
        logger.info(f"Session {session_id} connected ({mode} mode)")
        
        # Send connection confirmation
        await websocket.send_json({
            'type': 'connection',
            'status': 'connected',
            'session_id': session_id,
            'mode': mode,
            'recording': recording_status,
            'recording_path': wav_path
        })
        
    async def disconnect(self, session_id: str):
        """Disconnect session and cleanup"""
        websocket = self.active_connections.get(session_id)
        
        if session_id in self.engines:
            # Finalize session for improved engine (merge similar speakers)
            engine = self.engines[session_id]
            if hasattr(engine, 'finalize_session'):
                logger.info(f"Finalizing session {session_id}...")
                engine.finalize_session()
                
                # Send final summary with corrected speaker count and transcript
                if websocket and hasattr(engine, 'processed_segments'):
                    try:
                        final_speakers = set(seg.get('speaker_id') or seg.get('speaker') 
                                            for seg in engine.processed_segments)
                        
                        # Create serializable segments (exclude numpy arrays)
                        serializable_segments = []
                        for seg in engine.processed_segments:
                            serializable_segments.append({
                                'speaker': seg.get('speaker'),
                                'speaker_id': seg.get('speaker_id'),
                                'text': seg.get('text'),
                                'start_time': seg.get('start_time'),
                                'end_time': seg.get('end_time'),
                                'duration': seg.get('duration')
                            })
                        
                        await websocket.send_json({
                            'type': 'finalization',
                            'data': {
                                'speaker_count': len(final_speakers),
                                'speakers': sorted(list(final_speakers)),
                                'segments': serializable_segments
                            },
                            'timestamp': datetime.now().isoformat()
                        })
                        logger.info(f"Sent finalization summary: {len(final_speakers)} speakers")
                    except Exception as e:
                        logger.error(f"Failed to send finalization summary: {e}")
                        
            del self.engines[session_id]
            
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            
        if session_id in self.recorders:
            try:
                self.recorders[session_id].close()
            finally:
                del self.recorders[session_id]
                
        logger.info(f"Session {session_id} disconnected")
    
    async def process_audio_data(
        self,
        session_id: str,
        audio_data: bytes,
        mode: str = "single"
    ):
        """
        Process incoming audio data and return transcript if available.
        
        Args:
            session_id: Session identifier
            audio_data: Raw audio bytes (16-bit PCM, mono or stereo)
            mode: Processing mode
        """
        if session_id not in self.engines:
            logger.error(f"Session {session_id} not found")
            return
            
        engine = self.engines[session_id]
        websocket = self.active_connections[session_id]
        
        try:
            # Append raw PCM to session recorder first
            recorder = self.recorders.get(session_id)
            if recorder:
                recorder.write(audio_data)

            # Convert bytes to numpy array
            # Assuming 16-bit PCM audio
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0
            
            # Log audio statistics
            if len(audio_array) > 0:
                logger.debug(f"Session {session_id}: Received {len(audio_array)} samples, RMS: {np.sqrt(np.mean(audio_array**2)):.4f}")
            
            # Process based on mode
            if mode == "dual" and len(audio_array) % 2 == 0:
                # Stereo: split into left and right channels
                left_channel = audio_array[0::2]
                right_channel = audio_array[1::2]
                
                # Process both channels
                segments = await engine.process_stereo_chunk(left_channel, right_channel)
                
                # Send results for each segment
                for segment in segments:
                    await websocket.send_json({
                        'type': 'transcript',
                        'data': segment.to_dict(),
                        'timestamp': datetime.now().isoformat()
                    })
                    
            else:
                # Single channel
                segment = await engine.process_audio_chunk(audio_array)
                
                if segment is not None:
                    # Handle both dict and object returns
                    if isinstance(segment, dict):
                        speaker_id = segment.get('speaker_id') or segment.get('speaker')
                        text = segment.get('text', '')
                        segment_data = segment
                    else:
                        speaker_id = segment.speaker_id
                        text = segment.text
                        segment_data = segment.to_dict()
                    
                    logger.info(f"✓ SENDING TO CLIENT - {speaker_id}: {text}")
                    await websocket.send_json({
                        'type': 'transcript',
                        'data': segment_data,
                        'timestamp': datetime.now().isoformat()
                    })
                    logger.info(f"✓ Message sent successfully")
                    
        except Exception as e:
            logger.error(f"Error processing audio for session {session_id}: {e}", exc_info=True)
            await websocket.send_json({
                'type': 'error',
                'message': str(e)
            })
            
    async def get_conversation_history(self, session_id: str) -> Optional[list]:
        """Get conversation history for session"""
        if session_id in self.engines:
            return self.engines[session_id].get_conversation_history()
        return None
        
    # SOAP note generation commented out - focusing on speaker ID and transcription
    # async def generate_soap_note(self, session_id: str) -> Optional[dict]:
    #     """Generate SOAP note for session"""
    #     if session_id not in self.engines:
    #         return None
    #         
    #     engine = self.engines[session_id]
    #     segments = engine.get_conversation_history()
    #     
    #     if not segments:
    #         return {'error': 'No conversation data available'}
    #         
    #     # Generate SOAP note
    #     from app.summarizer_local import get_generator
    #     
    #     try:
    #         generator = get_generator()
    #         if hasattr(generator, 'generate_soap_note'):
    #             soap_note = generator.generate_soap_note(segments)
    #         else:
    #             # Fallback
    #             from app.summarizer_local import summarize_conversation
    #             soap_text = summarize_conversation(segments)
    #             soap_note = {'raw': soap_text}
    #             
    #         return soap_note
    #         
    #     except Exception as e:
    #         logger.error(f"SOAP generation failed: {e}")
    #         return {'error': str(e)}


# Global manager instance
stream_manager = AudioStreamManager()


async def handle_websocket_connection(
    websocket: WebSocket,
    session_id: str,
    mode: str = "single"
):
    """
    Main WebSocket handler for audio streaming.
    
    Protocol:
    - Client sends JSON with {'type': 'config', ...} for configuration
    - Client sends binary data for audio chunks
    - Client sends JSON with {'type': 'command', 'command': 'history'/'soap'/'reset'}
    - Server sends JSON with {'type': 'transcript'/'soap'/'error', ...}
    """
    await stream_manager.connect(websocket, session_id, mode)
    
    try:
        while True:
            # Receive data (can be JSON or binary)
            # First, check what type of message we're receiving
            message = await websocket.receive()
            
            # Handle binary audio data
            if 'bytes' in message:
                audio_bytes = message['bytes']
                await stream_manager.process_audio_data(session_id, audio_bytes, mode)
                
            # Handle text/JSON commands
            elif 'text' in message:
                try:
                    data = json.loads(message['text'])
                    
                    # Handle commands
                    if data.get('type') == 'command':
                        command = data.get('command')
                        
                        if command == 'history':
                            history = await stream_manager.get_conversation_history(session_id)
                            await websocket.send_json({
                                'type': 'history',
                                'data': history
                            })
                            
                        # SOAP generation commented out
                        # elif command == 'soap':
                        #     soap_note = await stream_manager.generate_soap_note(session_id)
                        #     await websocket.send_json({
                        #         'type': 'soap',
                        #         'data': soap_note
                        #     })
                            
                        elif command == 'reset':
                            if session_id in stream_manager.engines:
                                stream_manager.engines[session_id].reset()
                            await websocket.send_json({
                                'type': 'status',
                                'message': 'Session reset'
                            })
                            
                        else:
                            await websocket.send_json({
                                'type': 'error',
                                'message': f'Unknown command: {command}'
                            })
                            
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON message: {e}")
                except Exception as e:
                    logger.error(f"Error handling text message: {e}")
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
        await stream_manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for {session_id}: {e}")
        await stream_manager.disconnect(session_id)


class SessionWavRecorder:
    """Simple per-session WAV recorder that appends raw 16-bit PCM chunks."""
    def __init__(self, file_path: str, sample_rate: int, channels: int = 1):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.channels = channels
        self._lock = Lock()
        self._wf = wave.open(self.file_path, 'wb')
        self._wf.setnchannels(self.channels)
        self._wf.setsampwidth(2)  # 16-bit PCM
        self._wf.setframerate(self.sample_rate)
        logging.info(f"Recording to {self.file_path} ({self.channels}ch @ {self.sample_rate}Hz)")

    def write(self, pcm_bytes: bytes):
        if not pcm_bytes:
            return
        with self._lock:
            if self._wf:
                # writeframes handles byte alignment for sample width
                self._wf.writeframes(pcm_bytes)

    def close(self):
        with self._lock:
            if self._wf:
                try:
                    self._wf.close()
                finally:
                    self._wf = None
