# IU Learning Classroom
## Voice-Powered Interactive Education Platform

### 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run The Project**
   ```bash
   RUN_PROJECT.bat
   ```

3. **Access the Classroom**
   - Open your browser: http://127.0.0.1:8000
   - Select your role (Teacher/Student)
   - Start teaching or learning!

### 🎓 Features

- **Teacher Mode**: Upload lesson recordings or conduct live sessions
- **Student Mode**: Practice speaking and review lessons
- **Real-time Transcription**: Voice-to-text with speaker identification
- **Progress Tracking**: Monitor learning advancement
- **Interactive Classroom**: Full student-teacher interaction

### 📝 Enroll Speakers (Optional)

To improve speaker recognition:
```bash
venv\Scripts\python.exe scripts\enroll_speaker.py --id Teacher --audio teacher_sample.wav
venv\Scripts\python.exe scripts\enroll_speaker.py --id Student --audio student_sample.wav
```

### 🔧 Tech Stack

- FastAPI backend with WebSocket support
- SpeechBrain for ASR and speaker recognition
- Interactive HTML5 frontend
- Real-time audio streaming

---

**© 2026 IU Learning Classroom - Empowering Education Through Voice**
