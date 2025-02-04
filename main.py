# main.py

import eel
import soundcard as sc
import threading
import queue
import logging
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import numpy as np
import torch
import torch.nn as nn
import librosa
import ollama
import pythoncom  # Import pywin32's COM module
import json
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning, module='transformers')
warnings.filterwarnings("ignore", category=RuntimeWarning, module='soundcard')

# Constants
SETTINGS_FILE = 'settings.json'
SUPPORTED_LANGUAGES = [
    "English", "Chinese", "German", "Spanish", "Russian", "Korean", "French",
    "Japanese", "Portuguese", "Turkish", "Polish", "Catalan", "Dutch",
    "Arabic", "Swedish", "Italian", "Indonesian", "Hindi", "Finnish",
    "Vietnamese", "Hebrew", "Ukrainian", "Greek", "Malay", "Czech",
    "Romanian", "Danish", "Hungarian", "Tamil", "Norwegian", "Thai",
    "Urdu", "Croatian", "Bulgarian", "Lithuanian", "Latin", "Māori",
    "Malayalam", "Welsh", "Slovak", "Telugu", "Persian", "Latvian",
    "Bengali", "Serbian", "Azerbaijani", "Slovenian", "Kannada",
    "Estonian", "Macedonian", "Breton", "Basque", "Icelandic",
    "Armenian", "Nepali", "Mongolian", "Bosnian", "Kazakh", "Albanian",
    "Swahili", "Galician", "Marathi", "Panjabi", "Sinhala", "Khmer",
    "Shona", "Yoruba", "Somali", "Afrikaans", "Occitan", "Georgian",
    "Belarusian", "Tajik", "Sindhi", "Gujarati", "Amharic", "Yiddish",
    "Lao", "Uzbek", "Faroese", "Haitian", "Pashto", "Turkmen",
    "Norwegian Nynorsk", "Maltese", "Sanskrit", "Luxembourgish",
    "Burmese", "Tibetan", "Tagalog", "Malagasy", "Assamese", "Tatar",
    "Hawaiian", "Lingala", "Hausa", "Bashkir", "Jw"
]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Eel
eel.init('web')

# Function to pad or truncate mel features
def pad_or_truncate(input_features, target_length=3000):
    current_length = input_features.shape[-1]
    if current_length > target_length:
        input_features = input_features[..., :target_length]
    elif current_length < target_length:
        pad_width = target_length - current_length
        input_features = torch.nn.functional.pad(input_features, (0, pad_width))
    return input_features

def detect_silence(audio, samplerate, frame_duration=0.05, energy_threshold=0.05, silence_duration=0.5):
    frame_length = int(frame_duration * samplerate)
    hop_length = frame_length  # Non-overlapping frames

    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    silent_frames = energy < energy_threshold
    min_silence_frames = int(silence_duration / frame_duration)

    for i in range(len(silent_frames) - min_silence_frames, -1, -1):
        if all(silent_frames[i:i + min_silence_frames]):
            split_point = i * hop_length
            return split_point
    return None

class Settings:
    def __init__(self, filepath=SETTINGS_FILE):
        self.filepath = filepath
        self.default_settings = {
            "selected_device": "cpu",
            "source_language": "Spanish",
            "target_language": "English",
            "selected_audio_source": 0  # New setting added
        }
        self.settings = self.load_settings()

    def load_settings(self):
        if not os.path.exists(self.filepath):
            return self.default_settings.copy()
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                settings = json.load(f)
            # Ensure all default keys are present
            for key, value in self.default_settings.items():
                if key not in settings:
                    settings[key] = value
            return settings
        except Exception as e:
            logging.error(f"Error loading settings: {e}")
            return self.default_settings.copy()

    def save_settings(self):
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=4)
            logging.info("Settings saved successfully.")
        except Exception as e:
            logging.error(f"Error saving settings: {e}")

    def get_setting(self, key):
        return self.settings.get(key, self.default_settings.get(key))

    def set_setting(self, key, value):
        self.settings[key] = value
        self.save_settings()

class AudioRecorder(threading.Thread):
    def __init__(self, mic, source_type, samplerate=16000, chunk_duration=0.5, audio_queue=None):
        super().__init__()
        self.mic = mic
        self.source_type = source_type  # 'Microphone' or 'System Audio'
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)
        self.running = False
        self.audio_queue = audio_queue

    def run(self):
        pythoncom.CoInitialize()  # Initialize COM for this thread
        self.running = True
        try:
            with self.mic.recorder(samplerate=self.samplerate) as recorder:
                while self.running:
                    try:
                        data = recorder.record(numframes=self.chunk_size)
                        # If stereo, convert to mono
                        if len(data.shape) == 2 and data.shape[1] == 2:
                            data = data.mean(axis=1)
                        if self.audio_queue and not self.audio_queue.full():
                            self.audio_queue.put((data.copy(), self.source_type))
                    except sc.exceptions.SoundcardRuntimeWarning:
                        logging.warning("Data discontinuity in recording.")
                        continue
                    except Exception as e:
                        logging.error(f"Recording error: {e}")
                        eel.display_error(f"Recording error: {e}")
                        continue
        except Exception as e:
            logging.error(f"Failed to initialize recorder: {e}")
            eel.display_error(f"Failed to initialize recorder: {e}")
        finally:
            pythoncom.CoUninitialize()  # Uninitialize COM before thread exits

    def stop(self):
        self.running = False

class TranscriptionWorker(threading.Thread):
    def __init__(self, audio_queue, device, source_language, target_language, samplerate=16000):
        super().__init__()
        self.audio_queue = audio_queue
        self.running = False
        self.samplerate = samplerate
        self.device = device  # Device string, e.g., 'cpu', 'cuda'
        self.ollama_client = None
        self.source_language = source_language
        self.target_language = target_language

        # Parameters for processing
        self.CHUNK_LENGTH_S = 30  # 30 seconds
        self.OVERLAP_DURATION_S = 1  # 1 second overlap

        self.previous_transcription = ""

        # Initialize the Transformers model and processor
        try:
            model_id = "openai/whisper-large-v3-turbo"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            self.model.to(self.device)
            
            if self.device.startswith('cuda'):
                if torch.cuda.device_count() > 1:
                    logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
                    self.model = nn.DataParallel(self.model)
                else:
                    logging.info("Using single GPU.")
            else:
                logging.info("Using CPU for transcription.")
            
            self.model.eval()  # Set model to evaluation mode
            if self.device.startswith('cuda'):
                self.model.half()  # Use mixed precision if GPU is available

            self.ollama_client = ollama.Client()  # Adjust host if necessary
            logging.info("TranscriptionWorker initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization error: {e}")
            eel.display_error(f"Initialization error: {e}")
            self.model = None
            self.processor = None

    def generate(self, *args, **kwargs):
        """
        Helper method to call generate on the underlying model,
        handling DataParallel wrapping.
        """
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.generate(*args, **kwargs)
        else:
            return self.model.generate(*args, **kwargs)

    def run(self):
        self.running = True
        buffer = {}
        overlap_samples = int(self.OVERLAP_DURATION_S * self.samplerate)

        while self.running and self.model and self.processor:
            try:
                data, source_type = self.audio_queue.get(timeout=1)
                if source_type not in buffer:
                    buffer[source_type] = np.array([], dtype=np.float32)
                buffer[source_type] = np.concatenate((buffer[source_type], data))

                # First, attempt to detect silence
                split_point = detect_silence(
                    buffer[source_type],
                    self.samplerate,
                    frame_duration=0.05,
                    energy_threshold=0.02,
                    silence_duration=0.5
                )

                if split_point is not None and split_point > 0:
                    # Split at silence
                    segment = buffer[source_type][:split_point]
                    buffer[source_type] = buffer[source_type][split_point:]

                    if np.mean(np.abs(segment)) < 0.01:
                        continue  # Skip silent segments

                    self.process_segment(segment, source_type)
                else:
                    # No silence detected; check if buffer exceeds chunk length
                    if len(buffer[source_type]) >= (self.CHUNK_LENGTH_S + self.OVERLAP_DURATION_S) * self.samplerate:
                        # Extract chunk with overlap
                        chunk_end = self.CHUNK_LENGTH_S * self.samplerate
                        chunk = buffer[source_type][:chunk_end + overlap_samples]
                        buffer[source_type] = buffer[source_type][chunk_end:]

                        if np.mean(np.abs(chunk)) < 0.01:
                            continue  # Skip silent segments

                        self.process_segment(chunk, source_type)
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Processing error: {e}")
                eel.display_error(f"Processing error: {e}")

        # Process any remaining audio in the buffer when stopping
        for source_type, audio_data in buffer.items():
            if len(audio_data) > 0:
                self.process_segment(audio_data, source_type)

    def process_segment(self, segment, source_type):
        try:
            # Convert the segment to float32 if it's not
            if segment.dtype != np.float32:
                segment = segment.astype(np.float32)

            # Prepare input features
            with torch.no_grad():
                input_features = self.processor(segment, sampling_rate=self.samplerate, return_tensors="pt", padding="longest").input_features
                input_features = pad_or_truncate(input_features)
                input_features = input_features.to(self.device)
                if self.device.startswith('cuda'):
                    input_features = input_features.half()  # Convert to float16 for mixed precision

                if self.source_language.lower() in [lang.lower() for lang in SUPPORTED_LANGUAGES]:
                    language = self.source_language.lower()
                    forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
                else:
                    language = "auto"
                    forced_decoder_ids = None

                # Generate transcription
                predicted_ids = self.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_length=3000,
                    no_repeat_ngram_size=2,
                    early_stopping=True
                )
                transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Handle stitching to avoid duplication due to overlap
            if self.previous_transcription:
                # Approximate number of words to remove based on overlap duration
                overlap_fraction = self.OVERLAP_DURATION_S / self.CHUNK_LENGTH_S
                overlap_words = int(len(transcription.split()) * overlap_fraction)
                transcription_words = transcription.split()
                if len(transcription_words) > overlap_words:
                    transcription = ' '.join(transcription_words[overlap_words:])
                else:
                    transcription = ''

            # Update the previous transcription
            self.previous_transcription = transcription

            # Send the transcription to the frontend
            eel.update_raw_text(transcription)

            # Prepare prompt for translation
            prompt = (
                f"Read this quote: '{transcription}'"
                f"Translate the quote from {self.source_language} to {self.target_language} with correct grammar. "
                f"YOUR RESPONSE SHUOLD RETURN ONLY THE RAW DIRECT TRANSLTED QUOTE IN RESPONSE.  DO NOT PUT QUOTES AROUND IT ''. ONLY RETURN THE WORDS OF THE TRANSLATION"
            )
            target_type = self.target_language

            # Send prompt to Ollama for translation
            response = self.ollama_client.chat(
                model='llama3.2:3b-instruct-fp16',
                messages=[
                    {"role": "user", "content": prompt},
                ],
                stream=False  # Set to True if handling streaming responses
            )

            translated_text = response['message']['content'].strip()
            eel.update_translated_text(translated_text)
            logging.info(f"Translated text [{target_type}]: {translated_text}")

        except Exception as e:
            logging.error(f"Error during transcription/translation: {e}")
            eel.display_error(f"Error during transcription/translation: {e}")

    def stop(self):
        self.running = False

class TranscriptionApp:
    def __init__(self, settings):
        self.audio_queue = queue.Queue(maxsize=200)
        self.recorder_thread = None
        self.transcription_thread = None
        self.recording_devices = []
        self.available_devices = self.get_available_devices()
        self.settings = settings

    def get_available_devices(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')  # 'cuda' will utilize all available GPUs via DataParallel
        return devices

    def get_audio_sources(self):
        try:
            self.recording_devices = sc.all_microphones(include_loopback=True)
            sources = []
            for mic in self.recording_devices:
                device_name = mic.name.lower()
                if 'loopback' in device_name or 'stereo mix' in device_name or 'what u hear' in device_name:
                    label = f"System Audio - {mic.name}"
                    source_type = 'System Audio'
                else:
                    label = f"Microphone - {mic.name}"
                    source_type = 'Microphone'
                sources.append({'label': label, 'index': self.recording_devices.index(mic), 'type': source_type})
            eel.receive_audio_sources(sources)
            logging.info("Audio sources sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error fetching audio sources: {e}")
            eel.display_error(f"Error fetching audio sources: {e}")
            return False

    def get_device_list(self):
        try:
            devices = self.available_devices
            eel.receive_device_list(devices)
            logging.info("Device list sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error fetching device list: {e}")
            eel.display_error(f"Error fetching device list: {e}")
            return False

    def get_supported_languages(self):
        try:
            eel.receive_supported_languages(SUPPORTED_LANGUAGES)
            logging.info("Supported languages sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error sending supported languages: {e}")
            eel.display_error(f"Error sending supported languages: {e}")
            return False

    def start_transcription(self, selected_index, selected_device, source_language, target_language):
        if self.recorder_thread and self.recorder_thread.is_alive():
            eel.display_error("Transcription is already running.")
            return False

        try:
            mic = self.recording_devices[selected_index]
            source_type = 'System Audio' if ('loopback' in mic.name.lower() or 
                                            'stereo mix' in mic.name.lower() or 
                                            'what u hear' in mic.name.lower()) else 'Microphone'

            # Update settings
            self.settings.set_setting("selected_device", selected_device)
            self.settings.set_setting("source_language", source_language)
            self.settings.set_setting("target_language", target_language)
            self.settings.set_setting("selected_audio_source", selected_index)  # Save selected audio source

            self.recorder_thread = AudioRecorder(
                mic=mic,
                source_type=source_type,
                audio_queue=self.audio_queue
            )
            self.recorder_thread.start()
            logging.info("Recorder thread started.")

            if not self.transcription_thread or not self.transcription_thread.is_alive():
                self.transcription_thread = TranscriptionWorker(
                    self.audio_queue, 
                    selected_device, 
                    source_language, 
                    target_language
                )
                self.transcription_thread.start()
                logging.info("Transcription thread started.")

            eel.transcription_started()
            return True
        except Exception as e:
            logging.error(f"Error starting transcription: {e}")
            eel.display_error(f"Error starting transcription: {e}")
            return False

    def stop_transcription(self):
        try:
            if self.recorder_thread:
                self.recorder_thread.stop()
                self.recorder_thread.join()
                self.recorder_thread = None
                logging.info("Recorder thread stopped.")

            if self.transcription_thread:
                self.transcription_thread.stop()
                self.transcription_thread.join()
                self.transcription_thread = None
                logging.info("Transcription thread stopped.")

            eel.transcription_stopped()
            return True
        except Exception as e:
            logging.error(f"Error stopping transcription: {e}")
            eel.display_error(f"Error stopping transcription: {e}")
            return False

    def get_current_settings(self):
        try:
            eel.receive_current_settings(self.settings.settings)
            logging.info("Current settings sent to frontend.")
            return True
        except Exception as e:
            logging.error(f"Error sending current settings: {e}")
            eel.display_error(f"Error sending current settings: {e}")
            return False

# Instantiate Settings and TranscriptionApp
settings = Settings()
app = TranscriptionApp(settings)

# Define Eel-exposed functions outside the class
@eel.expose
def get_audio_sources():
    return app.get_audio_sources()

@eel.expose
def get_device_list():
    return app.get_device_list()

@eel.expose
def get_supported_languages():
    return app.get_supported_languages()

@eel.expose
def get_current_settings():
    return app.get_current_settings()

@eel.expose
def start_transcription(selected_index, selected_device, source_language, target_language):
    return app.start_transcription(selected_index, selected_device, source_language, target_language)

@eel.expose
def stop_transcription():
    return app.stop_transcription()

def main():
    eel.start(
        'index.html',
        mode='chrome',  # Specify the browser mode (e.g., 'chrome', 'edge', etc.)
        chrome_args=[
            '--resizable',               # Allow window resizing
            '--start-maximized'         # Start maximized
        ],
        block=False
    )
    app.get_audio_sources()
    app.get_device_list()
    app.get_supported_languages()
    app.get_current_settings()
    eel.sleep(1)  # Allow time for frontend to initialize

    # Keep the main thread alive
    try:
        while True:
            eel.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        logging.info("Shutting down application.")
        app.stop_transcription()

if __name__ == "__main__":
    main()
