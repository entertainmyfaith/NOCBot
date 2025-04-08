# all required libraries
import os
import base64
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sqlite3
import re
import librosa
import wave
import pyaudio
import time
import threading
import gc
import requests
from datetime import datetime

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from gtts import gTTS
import pygame

from twilio.twiml.voice_response import VoiceResponse, Dial, Say, Gather
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from twilio.rest import Client

# ===================== CONFIGURATION =====================
MODEL_TYPE = "openai/whisper-base"
LANGUAGE = "en"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
WAVE_OUTPUT_FILENAME = "recorded_audio.wav"
DATABASE_CONFIG = {
    'db_name': 'customers_db.sqlite',
    'table_name': 'Users'
}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Twilio Configuration
load_dotenv()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
OPERATOR_PHONE_NUMBER = os.getenv("OPERATOR_PHONE_NUMBER")

# ConnectWise Configuration
company_id = os.getenv("COMPANY_ID")
public_key = os.getenv("PUBLIC_KEY")
private_key = os.getenv("PRIVATE_KEY")
client_id = os.getenv("CLIENT_ID")
encoded_creds = os.getenv("ENCODED_CREDS")
CONNECTWISE_HEADERS = {
    'Content-Type': 'application/json',
    'clientId': client_id,
    'Authorization': f'{encoded_creds}'
}
CONNECTWISE_URL = "https://api-na.myconnectwise.net/v4_6_release/apis/3.0/"

# ===================== INITIAL SETUP =====================
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# ===================== AI MODELS =====================
class ChatbotModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ChatbotModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ===================== AUDIO COMPONENTS =====================
class AudioRecorder:
    def __init__(self):
        self.p = None
        self.stream = None
        self.frames = []
        self.is_recording = False
        self._init_audio()

    def _init_audio(self):
        if self.p:
            self.p.terminate()
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
            start=False
        )

    def record_audio(self):
        try:
            self.frames = []
            self.stream.start_stream()
            while self.is_recording:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as e:
            print(f"⚠️ Recording Error: {e}")
        finally:
            self.stream.stop_stream()

    def save_audio(self):
        with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(self.frames))

    def clean_up(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        self._init_audio()

class TextToSpeech:
    def __init__(self):
        pygame.mixer.init()

    def speak(self, text):
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save("speech.mp3")
            pygame.mixer.music.load("speech.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            pygame.mixer.music.unload()
            os.remove("speech.mp3")
        except Exception as e:
            print(f"⚠️ Text-to-Speech Error: {e}")

# ===================== NLP COMPONENTS =====================
class ChatbotAssistant:
    def __init__(self, intents_path):
        self.model = None
        self.intents_path = intents_path
        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = {}
        self.X = None
        self.y = None

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        return [lemmatizer.lemmatize(word.lower()) for word in words]

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)
        for intent in intents_data['intents']:
            self.intents.append(intent['tag'])
            self.intents_responses[intent['tag']] = intent['responses']
            for pattern in intent['patterns']:
                words = self.tokenize_and_lemmatize(pattern)
                self.vocabulary.extend(words)
                self.documents.append((words, intent['tag']))
        self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = [self.bag_of_words(doc[0]) for doc in self.documents]
        indices = [self.intents.index(doc[1]) for doc in self.documents]
        if not bags or not indices:
            raise ValueError("No training data found. Check your intents file.")
        self.X = np.array(bags)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)
        with open(dimensions_path, 'w') as f:
            json.dump({'input_size': self.X.shape[1], 'output_size': len(self.intents)}, f)

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensions = json.load(f)
        self.model = ChatbotModel(dimensions['input_size'], dimensions['output_size'])
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(words)
        bag_tensor = torch.tensor([bag], dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(bag_tensor)
        predicted_intent = self.intents[torch.argmax(predictions, dim=1).item()]
        return random.choice(self.intents_responses.get(predicted_intent,
                                                    ["I'm not sure how to respond to that."]))

def setup_chatbot():
    assistant = ChatbotAssistant('network_intents.json')
    assistant.parse_intents()
    assistant.prepare_data()
    if not os.path.exists('network_chatbot_model.pth'):
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)
        assistant.save_model('network_chatbot_model.pth', 'network_dimensions.json')
    else:
        assistant.load_model('network_chatbot_model.pth', 'network_dimensions.json')
    return assistant

# ===================== DATABASE COMPONENTS =====================
class CustomerManager:
    def __init__(self):
        self.db_conn = None
        self._init_db()
        self._verify_database()

    def get_db_connection(self):
        if self.db_conn is None:
            try:
                self.db_conn = sqlite3.connect(
                    DATABASE_CONFIG['db_name'],
                    check_same_thread=False,
                    timeout=10
                )
                self.db_conn.execute("PRAGMA foreign_keys = ON")
            except sqlite3.Error as e:
                print(f"⚠️ Database connection error: {e}")
                raise
        return self.db_conn

    def _init_db(self):
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {DATABASE_CONFIG['table_name']} (
                    account_number TEXT PRIMARY KEY,
                    customer_name TEXT NOT NULL,
                    account_type TEXT NOT NULL
                )
            """)
            conn.commit()
        except sqlite3.Error as e:
            print(f"⚠️ Database initialization error: {e}")
        finally:
            if conn:
                cursor.close()

    def _verify_database(self):
        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {DATABASE_CONFIG['table_name']}")
            count = cursor.fetchone()[0]
            print(f"\nDatabase contains {count} records")

            cursor.execute(f"SELECT * FROM {DATABASE_CONFIG['table_name']} LIMIT 5")
            results = cursor.fetchall()
            print("\nSample Records:")
            for row in results:
                print(f"Account: {row[0]}, Name: {row[1]}, Type: {row[2]}")
        except sqlite3.Error as e:
            print(f"⚠️ Database verification error: {e}")
        finally:
            if conn:
                cursor.close()

    def get_customer_info(self, account_number):
        if not account_number:
            return None

        conn = None
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            formatted_account = account_number.zfill(4)

            cursor.execute(
                f"SELECT customer_name, account_type FROM {DATABASE_CONFIG['table_name']} "
                "WHERE account_number = ?",
                (formatted_account,)
            )
            result = cursor.fetchone()
            return result
        except sqlite3.Error as e:
            print(f"⚠️ Database error in get_customer_info: {e}")
            return None
        finally:
            if conn:
                cursor.close()

    def __del__(self):
        if self.db_conn is not None:
            self.db_conn.close()

class WhisperTranscriber:
    def __init__(self):
        print("Initializing Whisper Transcriber...")
        self.processor = WhisperProcessor.from_pretrained(MODEL_TYPE)
        self.model = WhisperForConditionalGeneration.from_pretrained(MODEL_TYPE).to(DEVICE)

    def transcribe_audio(self, audio_file):
        try:
            audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
            input_features = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
            input_features = input_features.to(DEVICE)
            predicted_ids = self.model.generate(input_features)
            return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except Exception as e:
            print(f"⚠️ Transcription Error: {e}")
            return None

    def clean_up(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ===================== ConnectWise API Integration =====================
class ConnectWiseManager:
    def __init__(self):
        self.url = CONNECTWISE_URL
        self.headers = CONNECTWISE_HEADERS
        self.board_id = 1 
        self.company_id = 19297
        self.contact_id = 7610 
        self.status_id = 16 

    def create_ticket(self, description):
        """Creates a ticket with the exact structure requested"""
        try:
            ticket_data = {
                "summary": "Voice Assistant Ticket",
                "initialDescription": description,
                "board": {"id": int(self.board_id)},
                "company": {"id": int(self.company_id)},
                "contact": {"id": int(self.contact_id)},
                "status": {"id": int(self.status_id)},
            }

            response = requests.post(
                f"{self.url}service/tickets",
                headers=self.headers,
                json=ticket_data,
                timeout=10
            )

            if response.status_code == 201:
                return {"status": "success", "message": "Ticket created successfully"}
            return {"status": "failed", "message": f"API error: {response.text}"}

        except Exception as e:
            return {"status": "failed", "message": f"Error creating ticket: {str(e)}"}

# ===================== MAIN APPLICATION =====================
class TranscriptionApp:
    def __init__(self):
        self.audio_recorder = AudioRecorder()
        self.whisper_transcriber = WhisperTranscriber()
        self.customer_manager = CustomerManager()
        self.text_to_speech = TextToSpeech()
        self.chatbot_assistant = setup_chatbot()
        self.latest_transcription = None
        self.lock = threading.Lock()
        self.twilio_active = False
        self.customer_name = None
        self.customer_type = None
        self.verified_account_number = None
        self.connectwise_manager = ConnectWiseManager()
        self.conversation_log = []
        self.operator_keywords = [
            'operator', 'agent', 'representative', 'human', 'person', 
            'talk to someone', 'real person', 'live agent'
        ]
        self.exit_keywords = [
            'quit', 'goodbye', 'bye', 'end call', 'that\'s all', 'no more'
        ]

    def run(self):
        print("\n🤖 Starting the Network Troubleshooting Assistant...")
        self.start_flask()
        while True:
            time.sleep(1)

    def start_flask(self):
        app = Flask(__name__)

        @app.route('/twilio', methods=['POST'])
        def twilio_endpoint():
            self.twilio_active = True
            self.conversation_log = []  # Reset conversation log for new call
            response = VoiceResponse()
            response.say("Welcome to NOC Bot! I'm here to assist you with your network issues.", voice="Polly.Amy")
            gather = response.gather(
                input='speech',
                action='/get_account_number',
                timeout=5,
                speech_timeout='auto'
            )
            gather.say("Please say your 4-digit account number.", voice="Polly.Amy")
            response.say("Sorry, I didn't receive your account number. Goodbye.", voice="Polly.Amy")
            response.hangup()
            return str(response)

        @app.route('/get_account_number', methods=['POST'])
        def get_account_number():
            account_number_speech = request.form.get('SpeechResult', '')
            self.log_conversation("Customer", account_number_speech)
            account_number = self.extract_account_number(account_number_speech)
            response = VoiceResponse()
            
            if account_number:
                customer_info = self.customer_manager.get_customer_info(account_number)
                if customer_info:
                    name, acc_type = customer_info
                    verification_message = f"Thank you, {name}. I've verified your {acc_type} account."
                    response.say(verification_message, voice="Polly.Amy")
                    self.log_conversation("Bot", verification_message)
                    
                    self.customer_name = name
                    self.customer_type = acc_type.lower()  # Store as lowercase for easier comparison
                    self.verified_account_number = account_number
                    
                    # Ask about the problem
                    problem_prompt = "Please describe your network issue in detail."
                    response.say(problem_prompt, voice="Polly.Amy")
                    self.log_conversation("Bot", problem_prompt)
                    
                    gather = response.gather(
                        input='speech',
                        action='/handle_problem',
                        timeout=10,
                        speech_timeout='auto'
                    )
                else:
                    response.say("I'm sorry, I couldn't find your account. Please try again.", voice="Polly.Amy")
                    self.log_conversation("Bot", "Account not found response")
                    gather = response.gather(input='speech', action='/get_account_number', timeout=5, speech_timeout='auto')
                    gather.say("Please say your 4 digit account number.", voice="Polly.Amy")
            else:
                response.say("I couldn't understand your account number. Please try again.", voice="Polly.Amy")
                self.log_conversation("Bot", "Account number not understood")
                gather = response.gather(input='speech', action='/get_account_number', timeout=5, speech_timeout='auto')
                gather.say("Please say your 4 digit account number.", voice="Polly.Amy")

            return str(response)

        @app.route('/handle_problem', methods=['POST'])
        def handle_problem():
            problem_description = request.form.get('SpeechResult', '')
            self.log_conversation("Customer", problem_description)
            response = VoiceResponse()
            
            # Acknowledge the problem
            problem_lower = problem_description.lower()

            # Check if user is asking for an operator
            if any(keyword in problem_lower for keyword in self.operator_keywords):
                if self.customer_type == "business":
                    response.say("I'll connect you with an operator shortly. Please hold.", voice="Polly.Amy")
                    self.log_conversation("Bot", "Connecting to operator")
                    response.redirect('/end_conversation')
                    return str(response)
                else:
                    response.say("Our office is currently closed. I'll create a ticket and our team will contact you when we open.", voice="Polly.Amy")
                    self.log_conversation("Bot", "Office closed response")
                    response.redirect('/end_conversation')
                    return str(response)

            # Default chatbot response
            assistant_response = self.chatbot_assistant.process_message(problem_description)
            response.say(assistant_response, voice="Polly.Amy")
            self.log_conversation("Bot", assistant_response)
                        
            # Process the problem with the chatbot
            assistant_response = self.chatbot_assistant.process_message(problem_description)
            response.say(assistant_response, voice="Polly.Amy")
            self.log_conversation("Bot", assistant_response)
            
            # Ask if there's anything else
            follow_up = "Is there anything else I can help you with?"
            response.say(follow_up, voice="Polly.Amy")
            self.log_conversation("Bot", follow_up)
            
            gather = response.gather(
                input='speech',
                action='/process_query',
                timeout=10,
                speech_timeout='auto'
            )
            
            return str(response)

        @app.route('/process_query', methods=['POST'])
        def process_query_route():
            query_speech = request.form.get('SpeechResult', '')
            self.log_conversation("Customer", query_speech)
            response = self.process_query(query_speech)
            self.twilio_active = False
            return str(response)

        @app.route('/end_conversation', methods=['POST'])
        def end_conversation():
            response = VoiceResponse()
            # Create ticket with full conversation log
            ticket_description = self.generate_ticket_description()
            ticket_response = self.connectwise_manager.create_ticket(ticket_description)
            
            if ticket_response["status"] == "success":
                if self.customer_type == "business":
                    # Notify operator for business customers
                    operator_message = f"Business customer {self.customer_name} (Account: {self.verified_account_number}) needs assistance. Full conversation logged in ticket."
                    self.send_sms(operator_message)
                    response.say("I've created a ticket for your issue and notified our team. They will contact you shortly. Thank you for calling!", voice="Polly.Amy")
                else:
                    # Residential customer message
                    response.say("I've created a ticket for your issue. Our support team will contact you when the office opens. Thank you for calling!", voice="Polly.Amy")
            else:
                response.say("I couldn't create a ticket for your issue. Please call back later.", voice="Polly.Amy")
            
            response.hangup()
            return str(response)

        def run_flask():
            app.run(host='0.0.0.0', port=5000, use_reloader=False)

        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()

    def log_conversation(self, speaker, text):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.conversation_log.append(f"{timestamp} - {speaker}: {text}")

    def generate_ticket_description(self):
        description = f"Customer: {self.customer_name}\n"
        description += f"Account Number: {self.verified_account_number}\n"
        description += f"Account Type: {self.customer_type}\n"
        description += "Conversation Log:\n"
        description += "\n".join(self.conversation_log)
        return description

    def extract_account_number(self, speech_text):
        account_match = re.search(r'\b(\d{4})\b', speech_text)
        if account_match:
            return account_match.group(1)
        else:
            account_match = re.search(r'(\d{4})', speech_text)
            if account_match:
                return account_match.group(1)
            return None

    def process_query(self, transcription):
        response = VoiceResponse()
        if not transcription:
            response.say("I didn't catch that. Could you please repeat?", voice="Polly.Amy")
            self.log_conversation("Bot", "Didn't catch response")
            gather = response.gather(
                input='speech',
                action='/process_query',
                timeout=10,
                speech_timeout='auto'
            )
            return str(response)
            
        transcription_lower = transcription.lower()
        
        # Check for exit conditions
        if any(word in transcription_lower for word in self.exit_keywords):
            response.redirect('/end_conversation')
            return str(response)
            
        # Check if customer is asking for an operator
        if any(keyword in transcription_lower for keyword in self.operator_keywords):
            if self.customer_name and self.customer_type:
                # For business customers, we'll notify operator when creating ticket
                if self.customer_type == "business":
                    response.say("I'll connect you with an operator shortly. Please hold.", voice="Polly.Amy")
                    self.log_conversation("Bot", "Connecting to operator")
                    # We'll handle the operator notification in the end_conversation flow
                    response.redirect('/end_conversation')
                else:
                    response.say("Our office is currently closed. I'll create a ticket and our team will contact you when we open.", voice="Polly.Amy")
                    self.log_conversation("Bot", "Office closed response")
                    response.redirect('/end_conversation')
            else:
                response.say("Please provide your account number first.", voice="Polly.Amy")
                self.log_conversation("Bot", "Account number required")
            return str(response)
            
        # Default chatbot response
        assistant_response = self.chatbot_assistant.process_message(transcription)
        response.say(assistant_response, voice="Polly.Amy")
        self.log_conversation("Bot", assistant_response)
        
        follow_up = "Is there anything else I can help you with?"
        response.say(follow_up, voice="Polly.Amy")
        self.log_conversation("Bot", follow_up)
        
        gather = response.gather(
            input='speech',
            action='/process_query',
            timeout=10,
            speech_timeout='auto'
        )
        return str(response)

    def send_sms(self, message):
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        try:
            message = client.messages.create(
                body=message,
                from_=TWILIO_PHONE_NUMBER,
                to=OPERATOR_PHONE_NUMBER
            )
            print(f"SMS sent: {message.sid}")
            self.log_conversation("System", f"SMS sent to operator: {message}")
        except Exception as e:
            print(f"Error sending SMS: {e}")
            self.log_conversation("System", f"SMS failed: {str(e)}")

    def clean_up(self):
        self.audio_recorder.clean_up()
        self.whisper_transcriber.clean_up()
        gc.collect()

if __name__ == "__main__":
    print(f"Starting application on {DEVICE.upper()} device")
    app = TranscriptionApp()
    app.run()
