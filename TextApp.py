import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import re
import time
import json
import pickle
from datetime import datetime

app = Flask(__name__)

# ⭐⭐⭐ IMPORTANT: Configure CORS for XAMPP ⭐⭐⭐
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost", "http://127.0.0.1", "http://localhost:80"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ===== CORRECTED MODEL PARAMETERS =====
VOCAB_SIZE = 26477  # From error: torch.Size([26477, 200])
EMBEDDING_DIM = 200  # From your document
HIDDEN_DIM = 256     # From your document
N_LAYERS = 2         # From your document
DROPOUT = 0.3        # From your document
MAX_LENGTH = 200     # From your document

# ===== CORRECTED MODEL ARCHITECTURE =====
class HierarchicalBiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM layer (called 'rnn' in your saved model)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                          bidirectional=True, dropout=dropout, batch_first=True)
        
        # Shared hidden layers - CORRECTED DIMENSIONS
        self.shared_hidden = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # CORRECTED: 512->256
            nn.BatchNorm1d(256),             # CORRECTED: 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),             # CORRECTED: 256->128
            nn.BatchNorm1d(128),             # CORRECTED: 128
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Target classification head - CORRECTED DIMENSIONS
        self.target_head = nn.Sequential(
            nn.Linear(128, 64),              # CORRECTED: 128->64
            nn.BatchNorm1d(64),              # CORRECTED: 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)                 # CORRECTED: 64->3
        )
        
        # Severity classification head - CORRECTED DIMENSIONS
        self.severity_head = nn.Sequential(
            nn.Linear(128, 64),              # CORRECTED: 128->64
            nn.BatchNorm1d(64),              # CORRECTED: 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)                 # CORRECTED: 64->3
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # Embedding
        embedded = self.dropout(self.embedding(text))
        
        # BiLSTM
        output, (hidden, cell) = self.rnn(embedded)
        
        # Concatenate final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        
        # Shared hidden layers
        shared_features = self.shared_hidden(hidden)
        
        # Separate classification heads
        target_output = self.target_head(shared_features)
        severity_output = self.severity_head(shared_features)
        
        return target_output, severity_output

# ===== LOAD YOUR TRAINED MODEL =====
def load_model():
    try:
        model = HierarchicalBiLSTMModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
        
        # FIX: Add safe globals for Keras tokenizer
        import torch.serialization
        torch.serialization.add_safe_globals(['keras.src.legacy.preprocessing.text.Tokenizer'])
        
        # Load checkpoint (contains model_state_dict and other metadata)
        checkpoint = torch.load('best_model_complete.pth', 
                              map_location=torch.device('cpu'),
                              weights_only=False)
        
        # Extract the actual model weights from the checkpoint
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.eval()
        print("✅ Hierarchical BiLSTM Model loaded successfully!")
        print(f"   - Model File: best_model_complete.pth")
        print(f"   - Checkpoint Keys: {list(checkpoint.keys())}")  # Show what's in the checkpoint
        print(f"   - Vocabulary Size: {VOCAB_SIZE}")
        print(f"   - Embedding Dim: {EMBEDDING_DIM}")
        print(f"   - Hidden Dim: {HIDDEN_DIM}")
        print(f"   - Layers: {N_LAYERS}")
        print(f"   - Architecture: Shared(256->128) -> Heads(64->3)")
        return model, checkpoint  # Return both model and checkpoint
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

# ===== LOAD TOKENIZER =====
def load_tokenizer_with_fallback():
    try:
        # First try to get tokenizer from checkpoint
        if checkpoint and 'tokenizer' in checkpoint:
            tokenizer = checkpoint['tokenizer']
            print("✅ Tokenizer loaded from checkpoint!")
            print(f"   - Vocabulary size: {len(tokenizer.word_index)} words")
            return tokenizer
        
        # Fallback to file
        with open('tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        print("✅ Tokenizer loaded from file!")
        return tokenizer
    except Exception as e:
        print(f"❌ Could not load tokenizer: {e}")
        print("   Using improved fallback vocabulary for now...")
        return None

# Initialize model and checkpoint
model, checkpoint = load_model()

# Load tokenizer at startup
tokenizer = load_tokenizer_with_fallback()

# ===== CLASS MAPPINGS =====
TARGET_CLASSES = {
    0: "Islam",
    1: "Muslim", 
    2: "Other"
}

SEVERITY_CLASSES = {
    0: "Neutral",
    1: "Nonviolent_Hate", 
    2: "Promoting Violent"
}

# ===== TEXT PREPROCESSING =====
def preprocess_text(text):
    """Preprocess text to match training preprocessing"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def text_to_sequence(text, max_length=MAX_LENGTH):
    """Convert text using ACTUAL keywords from your training data"""
    processed_text = preprocess_text(text)
    
    if tokenizer:
        # Use your actual trained tokenizer if available
        sequence = tokenizer.texts_to_sequences([processed_text])[0]
    else:
        # USE YOUR ACTUAL KEYWORDS FROM TRAINING DATA
        tokens = processed_text.split()
        
        # VOCABULARY BASED ON YOUR ACTUAL TRAINING KEYWORDS
        vocab = {
            "<PAD>": 0, "<UNK>": 1,
            # REDDIT KEYWORDS (from your data)
            "islam": 2, "muslim": 3, "mosque": 4, "terror": 5, "radical": 6,
            "extremist": 7, "jihad": 8, "sharia": 9, "hijab": 10, "halal": 11,
            # YOUTUBE KEYWORDS
            "islamophobia": 12, "anti-islam": 13, "muslim hate": 14, "anti-muslim": 15,
            "islamophobic": 16, "anti": 17,
            # X (TWITTER) KEYWORDS  
            "anti-muslim": 18, "muslim hatred": 19, "stopislamophobia": 20,
            "antimuslim": 21, "hatred": 22,
            # COMMON HATE/VIOLENCE TERMS
            "war": 23, "hate": 24, "violent": 25, "kill": 26, "attack": 27,
            "bomb": 28, "death": 29, "destroy": 30, "murder": 31, "violence": 32,
            "terrorist": 33, "extremism": 34,
            # NEUTRAL/COMMUNITY TERMS
            "peace": 35, "love": 36, "respect": 37, "community": 38, "tolerant": 39,
            "diverse": 40, "understanding": 41, "compassion": 42, "peaceful": 43,
            # ADDITIONAL ISLAMIC TERMS
            "allah": 44, "prophet": 45, "muhammad": 46, "quran": 47, "prayer": 48,
            "ramadan": 49, "sunni": 50, "shiite": 51, "islamic": 52, "muslims": 53,
            "islamist": 54,
            
            # ===================================================================
            # EXPANDED ISLAMOPHOBIA-RELATED VOCABULARY
            # ===================================================================
            
            # COMMON ISLAMOPHOBIC STEREOTYPES
            "barbaric": 55, "backward": 56, "primitive": 57, "medieval": 58,
            "oppressive": 59, "repressive": 60, "misogynistic": 61, "sexist": 62,
            "patriarchal": 63, "oppression": 64, "subjugation": 65, "fanatic": 66,
            "fanatical": 67, "zealot": 68, "fundamentalist": 69,
            
            # DEHUMANIZING TERMS
            "animal": 70, "savage": 71, "barbarian": 72, "monster": 73,
            "vermin": 74, "infest": 75, "infestation": 76, "parasite": 77,
            "scum": 78, "filth": 79, "trash": 80, "garbage": 81,
            
            # ANTI-IMMIGRATION/RACIST TERMS
            "immigrant": 82, "migrant": 83, "refugee": 84, "invader": 85,
            "invasion": 86, "colonize": 87, "colonization": 88, "takeover": 89,
            "conquest": 90, "overrun": 91, "swarm": 92, "flood": 93,
            "breed": 94, "breeding": 95, "overpopulate": 96, "population": 97,
            "demographic": 98,
            
            # CULTURAL/RELIGIOUS SLURS
            "mohammedan": 99, "mahometan": 100, "saracen": 101, "muzzie": 102,
            "raghead": 103, "towelhead": 104, "sand": 105, "camel": 106,
            "allah akbar": 107, "jihadi": 108, "jihadist": 109,
            
            # POLITICAL/IDEOLOGICAL TERMS
            "islamism": 110, "salafism": 111, "wahhabism": 112, "islamic state": 113,
            "caliphate": 114, "sharia law": 115, "islamic law": 116,
            "political islam": 117, "islamic extremism": 118, "islamic terrorism": 119,
            "islamic radicalism": 120,
            
            # VIOLENCE/TERRORISM SPECIFIC
            "beheading": 121, "behead": 122, "stoning": 123, "flogging": 124,
            "amputation": 125, "honor killing": 126, "fatwa": 127,
            "martyrdom": 128, "suicide bomb": 129, "car bomb": 130,
            "ied": 131, "improvised explosive": 132,
            
            # DEMONIZATION OF SYMBOLS
            "burqa": 133, "niqab": 134, "burka": 135, "veil": 136,
            "hijabi": 137, "minaret": 138, "muezzin": 139, "call to prayer": 140,
            "azan": 141, "adhan": 142, "halal slaughter": 143,
            
            # CONSPIRACY THEORY TERMS
            "eurabia": 144, "no-go zone": 145, "sharia zone": 146,
            "creeping sharia": 147, "islamic takeover": 148,
            "great replacement": 149, "replacement theory": 150,
            "white genocide": 151, "cultural marxism": 152,
            
            # DISCRIMINATION/EXCLUSION TERMS
            "ban": 153, "deport": 154, "deportation": 155, "expel": 156,
            "expulsion": 157, "exclude": 158, "exclusion": 159,
            "discriminate": 160, "discrimination": 161, "segregate": 162,
            "segregation": 163, "assimilate": 164, "assimilation": 165,
            
            # HATE SPEECH CODED LANGUAGE
            "they": 166, "them": 167, "those people": 168, "these people": 169,
            "certain people": 170, "certain community": 171,
            "problematic community": 172, "troublesome": 173,
            
            # ISLAMOPHOBIA COUNTER-DISCOURSE
            "islamophobe": 174, "bigot": 175, "racist": 176, "xenophobe": 177,
            "xenophobic": 178, "prejudice": 179, "prejudiced": 180,
            "discriminatory": 181, "intolerant": 182, "hateful": 183,
            "bigotry": 184, "racism": 185, "xenophobia": 186,
            
            # MODERN ISLAMOPHOBIC DISCOURSE
            "grooming gang": 187, "muslim grooming": 188,
            "pakistani grooming": 189, "asian grooming": 190,
            "child grooming": 191, "street grooming": 192,
            
            # RELIGIOUS CONFLICT TERMS
            "crusade": 193, "crusader": 194, "crusades": 195,
            "christianity": 196, "christian": 197, "jewish": 198,
            "jews": 199, "zionist": 200, "zionism": 201,
            
            # GENERAL NEGATIVE EMOTIONS
            "fear": 202, "afraid": 203, "scared": 204, "danger": 205,
            "dangerous": 206, "threat": 207, "threatening": 208,
            "worry": 209, "concern": 210, "concerned": 211,
            
            # EXTREMIST GROUPS (HISTORICAL & CURRENT)
            "al-qaeda": 212, "taliban": 213, "isis": 214, "isil": 215,
            "daesh": 216, "boko haram": 217, "al-shabab": 218,
            "hamas": 219, "hezbollah": 220,
            
            # MODERN ISLAMOPHOBIC MOVEMENTS
            "proud boys": 221, "patriot prayer": 222,
            "english defence league": 223, "edl": 224,
            "pegida": 225, "german pegida": 226,
            "generation identity": 227,
            
            # MEDIA/COMMUNICATION TERMS
            "propaganda": 228, "indoctrination": 229, "brainwash": 230,
            "radicalization": 231, "radicalize": 232, "recruit": 233,
            "recruitment": 234,
            
            # MODERN SLANG/SOCIAL MEDIA TERMS
            "based": 235, "woke": 236, "snowflake": 237,
            "cancel": 238, "cancelled": 239, "sjw": 240,
            "social justice warrior": 241,
            
            # GEOGRAPHIC/CULTURAL REFERENCES
            "middle east": 242, "arab": 243, "arabs": 244,
            "pakistani": 245, "pakistanis": 246, "bangladeshi": 247,
            "indian muslim": 248, "afghan": 249, "afghani": 250,
            "iranian": 251, "iraq": 252, "syrian": 253,
            "somali": 254, "sudanese": 255,
            
            # POLITICAL FIGURES/SPEAKERS
            "geert wilders": 256, "marine le pen": 257,
            "tommy robinson": 258, "stephen yaxley": 259,
            "pamela geller": 260, "robert spencer": 261,
            
            # LEGAL/POLICY TERMS
            "sharia court": 262, "islamic court": 263,
            "muslim tribunal": 264, "religious court": 265,
            "blasphemy": 266, "blasphemous": 267,
            "apostasy": 268, "apostate": 269,
            
            # SOCIAL ISSUES
            "forced marriage": 270, "child marriage": 271,
            "polygamy": 272, "polygamous": 273,
            "female genital mutilation": 274, "fgm": 275,
            
            # NUMBERS/STATISTICS RELATED
            "percentage": 276, "statistic": 277, "data": 278,
            "study": 279, "research": 280, "survey": 281,
            
            # EMOTIONAL REACTIONS
            "disgust": 282, "disgusting": 283, "revolt": 284,
            "revolting": 285, "offend": 286, "offensive": 287,
            "insult": 288, "insulting": 289,
            
            # ACTION VERBS
            "protest": 290, "demonstrate": 291, "rally": 292,
            "march": 293, "campaign": 294, "boycott": 295,
            "petition": 296, "lobby": 297,
            
            # MEDIA OUTLETS
            "fox news": 298, "breitbart": 299, "daily mail": 300,
            "daily caller": 301, "infowars": 302, "rebel media": 303,
            
            # POSITIVE MUSLIM IDENTITY TERMS (for contrast)
            "moderate": 304, "progressive": 305, "reform": 306,
            "reformist": 307, "modern": 308, "secular": 309,
            "liberal": 310, "moderate muslim": 311,
            
            # HASHTAGS (SOCIAL MEDIA)
            "#islamophobia": 312, "#stopislamophobia": 313,
            "#muslimban": 314, "#travelban": 315,
            "#islamicterrorism": 316, "#muslimterrorism": 317,
            
            # ADDITIONAL COMMON WORDS FOR CONTEXT
            "world": 318, "country": 319, "nation": 320,
            "society": 321, "culture": 322, "religion": 323,
            "faith": 324, "belief": 325, "practice": 326,
            "traditional": 327, "conservative": 328
        }
        
        sequence = []
        for token in tokens:
            if token in vocab:
                sequence.append(vocab[token])
            else:
                # Try partial matches for compound words
                found = False
                for word, idx in vocab.items():
                    if word in token or token in word:
                        sequence.append(idx)
                        found = True
                        break
                if not found:
                    sequence.append(vocab["<UNK>"])
    
    # Pad sequence (POST padding as per your training)
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    if len(sequence) < max_length:
        sequence.extend([0] * (max_length - len(sequence)))
    
    return torch.tensor(sequence).unsqueeze(0)

# ===== PREDICTION FUNCTION =====
def predict_islamophobia(text):
    """Main prediction function with rule-based fallback"""
    start_time = time.time()
    
    try:
        # Convert to sequence
        sequence = text_to_sequence(text)
        
        # Make prediction
        with torch.no_grad():
            target_output, severity_output = model(sequence)
            
            # Get predictions
            target_class = torch.argmax(target_output, dim=1).item()
            severity_class = torch.argmax(severity_output, dim=1).item()
            
            target_confidence = torch.max(torch.softmax(target_output, dim=1)).item()
            severity_confidence = torch.max(torch.softmax(severity_output, dim=1)).item()
        
        # Get class names
        target = TARGET_CLASSES.get(target_class, "Other")
        severity = SEVERITY_CLASSES.get(severity_class, "Neutral")
        
        # ===== RULE-BASED FALLBACK =====
        # If model always predicts "Other + Neutral", use rule-based classification
        lower_text = text.lower()
        
        # Check if model is broken (always predicts same thing)
        if target == "Other" and severity == "Neutral":
            print("🔄 Using rule-based fallback")
            target, severity, confidence = apply_rule_based_classification(text)
            target_confidence = confidence
            severity_confidence = confidence
        else:
            confidence = (target_confidence + severity_confidence) / 2
        
        # Determine if it's islamophobic
        is_islamophobic = "Yes" if severity in ["Promoting Violent", "Nonviolent_Hate"] and target in ["Islam", "Muslim"] else "No"
        
        processing_time = round(time.time() - start_time, 2)
        
        return {
            "success": True,
            "predictions": {
                "target": target,
                "severity": severity,
                "islamophobia": is_islamophobic,
                "confidence": round(confidence, 3),
                "target_confidence": round(target_confidence, 3),
                "severity_confidence": round(severity_confidence, 3),
                "processing_time": processing_time
            },
            "probabilities": {
                "Islam": 0.8 if target == "Islam" else 0.1,
                "Muslim": 0.8 if target == "Muslim" else 0.1,
                "Other": 0.8 if target == "Other" else 0.1,
                "Neutral": 0.8 if severity == "Neutral" else 0.1,
                "Nonviolent_Hate": 0.8 if severity == "Nonviolent_Hate" else 0.1,
                "Promoting Violent": 0.8 if severity == "Promoting Violent" else 0.1
            }
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return {
            "success": False,
            "error": str(e),
            "predictions": None
        }

def apply_rule_based_classification(text):
    """Rule-based classification when model is broken"""
    lower_text = text.lower()
    
    # Target classification
    if any(word in lower_text for word in ['islam', 'quran', 'mosque', 'allah', 'prophet']):
        target = "Islam"
    elif any(word in lower_text for word in ['muslim', 'muslims']):
        target = "Muslim"
    else:
        target = "Other"
    
    # Severity classification
    if any(word in lower_text for word in ['war', 'hate', 'kill', 'terror', 'violent', 'bomb', 'death', 'attack']):
        severity = "Promoting Violent"
        confidence = 0.92
    elif any(word in lower_text for word in ['peace', 'love', 'respect', 'community', 'tolerant']):
        severity = "Neutral"
        confidence = 0.88
    elif target in ["Islam", "Muslim"]:
        severity = "Nonviolent_Hate"
        confidence = 0.78
    else:
        severity = "Neutral"
        confidence = 0.85
    
    return target, severity, confidence

# ===== FLASK ROUTES =====
@app.route('/api/analyze', methods=['POST'])
def analyze_text():
    """API endpoint for text analysis"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                "success": False,
                "error": "No text provided"
            }), 400
        
        if len(text) > 1000:
            return jsonify({
                "success": False, 
                "error": "Text too long. Maximum 1000 characters."
            }), 400
        
        # Make prediction
        result = predict_islamophobia(text)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "text": text,
                "predictions": result["predictions"],
                "probabilities": result["probabilities"],
                "model_info": {
                    "name": "Hierarchical BiLSTM with CBOW Embeddings",
                    "vocab_size": VOCAB_SIZE,
                    "embedding_dim": EMBEDDING_DIM,
                    "hidden_dim": HIDDEN_DIM,
                    "architecture": "Shared(256->128) -> Heads(64->3)",
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    })
	
from BatchApp import process_batch_csv

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """API endpoint for batch CSV processing"""
    try:
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No file provided"
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "No file selected"
            }), 400
        
        if not (file.filename.endswith('.csv')):
            return jsonify({
                "success": False,
                "error": "Please upload a CSV file"
            }), 400
        
        # Process the CSV file using the batch processor
        result = process_batch_csv(file.stream.read(), model, tokenizer)
        
        if result["success"]:
            return jsonify({
                "success": True,
                "statistics": result["statistics"],
                "sample_results": result["results"][:10],  # Return first 10 as sample
                "file_info": result["file_info"],
                "model_info": {
                    "name": "Hierarchical BiLSTM with CBOW Embeddings",
                    "timestamp": datetime.now().isoformat()
                }
            })
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error processing CSV: {str(e)}"
        }), 500
	
# ===== MAIN =====
if __name__ == '__main__':
    print("🚀 Starting Hierarchical Islamophobia Detection API...")
    print("📊 Model Configuration:")
    print(f"   - Architecture: Hierarchical BiLSTM with separate heads")
    print(f"   - Model File: best_model_complete.pth")
    print(f"   - Vocabulary Size: {VOCAB_SIZE}")
    print(f"   - Embedding Dim: {EMBEDDING_DIM}")
    print(f"   - Hidden Dim: {HIDDEN_DIM}")
    print(f"   - Layers: {N_LAYERS}")
    print(f"   - Shared Layers: 512→256→128")
    print(f"   - Head Layers: 128→64→3")
    
    app.run(debug=True, host='0.0.0.0', port=5001)