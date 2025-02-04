# Real-Time Speech Transcription & Translation App

This project is a **real-time speech transcription and translation application** that leverages **OpenAI's Whisper model** and **Ollama** for multilingual speech recognition and translation. It uses **Eel** to provide a web-based user interface, making it accessible and easy to use.

### Features:
- **Speech-to-Text Transcription**: Uses OpenAI’s Whisper model for accurate multilingual speech recognition.
- **Real-Time Audio Processing**: Captures live audio from system or microphone and transcribes it on the fly.
- **Automatic Language Translation**: Translates transcribed text into a target language using Ollama.
- **GPU Acceleration Support**: Utilizes CUDA for faster inference when available.
- **Configurable Settings**: Allows users to select input sources, processing devices (CPU/GPU), and language preferences.
- **Web-Based UI**: Built with Eel for a lightweight frontend.

### Technologies Used:
- **Python**: Core programming language
- **Eel**: Web-based UI framework
- **Transformers (Hugging Face)**: For speech processing
- **Torch**: Backend for model execution
- **Librosa**: Audio analysis and silence detection
- **Ollama**: Translation and text processing
- **SoundCard**: Capturing audio from different sources

### Setup & Usage:
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Application**:
   ```bash
   python main.py
   ```
3. **Check GPU Compatibility**:
   ```bash
   python gpu_test.py
   ```

### Future Enhancements:
- Support for additional speech models
- Improved UI and user experience
- Enhanced real-time translation accuracy
