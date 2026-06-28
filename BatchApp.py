import pandas as pd
import io
import torch
import re
import numpy as np
from datetime import datetime

# ===== CONSTANTS =====
MAX_LENGTH = 200
TARGET_CLASSES = {0: "Islam", 1: "Muslim", 2: "Other"}
SEVERITY_CLASSES = {0: "Neutral", 1: "Nonviolent_Hate", 2: "Promoting Violent"}

# ===== BATCH PROCESSING FUNCTIONS =====
def process_batch_csv(file_stream, model, tokenizer, max_records=100):
    """
    Process CSV file in batches using the actual model
    """
    try:
        # Read CSV file
        csv_data = pd.read_csv(io.StringIO(file_stream.decode("utf-8")))
        
        # Find text column
        text_column = find_text_column(csv_data)
        
        if not text_column:
            return {
                "success": False,
                "error": "No suitable text column found in CSV file"
            }
        
        texts = csv_data[text_column].dropna().astype(str).tolist()
        
        if len(texts) == 0:
            return {
                "success": False,
                "error": "No text data found in CSV file"
            }
        
        # Limit to max_records for performance
        if len(texts) > max_records:
            texts = texts[:max_records]
        
        # Process all texts
        results = []
        for i, text in enumerate(texts):
            try:
                result = predict_single_text(text, model, tokenizer)
                if result["success"]:
                    # Fix severity format (remove underscore for display)
                    severity = result["predictions"]["severity"]
                    if severity == "Nonviolent_Hate":
                        display_severity = "Nonviolent Hate"
                    else:
                        display_severity = severity
                    
                    results.append({
                        "text": truncate_text(text),
                        "full_text": text,
                        "target": result["predictions"]["target"],
                        "severity": severity,  # Keep original for consistency
                        "display_severity": display_severity,
                        "islamophobia": result["predictions"]["islamophobia"],
                        "confidence": result["predictions"]["confidence"],
                        "target_confidence": result["predictions"]["target_confidence"],
                        "severity_confidence": result["predictions"]["severity_confidence"]
                    })
                else:
                    # Fallback to rule-based
                    target, severity, confidence = rule_based_fallback(text)
                    if severity == "Nonviolent_Hate":
                        display_severity = "Nonviolent Hate"
                    else:
                        display_severity = severity
                    
                    results.append({
                        "text": truncate_text(text),
                        "full_text": text,
                        "target": target,
                        "severity": severity,
                        "display_severity": display_severity,
                        "islamophobia": "Yes" if severity in ["Promoting Violent", "Nonviolent_Hate"] and target in ["Islam", "Muslim"] else "No",
                        "confidence": confidence,
                        "target_confidence": confidence,
                        "severity_confidence": confidence
                    })
                    
            except Exception as e:
                print(f"Error processing text {i}: {e}")
                continue
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        return {
            "success": True,
            "statistics": stats,
            "results": results,
            "file_info": {
                "total_records": len(texts),
                "processed_records": len(results),
                "text_column": text_column,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        print(f"Error in process_batch_csv: {e}")
        return {
            "success": False,
            "error": f"Error processing CSV: {str(e)}"
        }

def find_text_column(df):
    """Find the text column in the DataFrame"""
    text_columns = ['text', 'Text', 'content', 'Content', 'tweet', 'Tweet', 
                   'message', 'Message', 'comment', 'Comment', 'post', 'Post',
                   'review', 'Review', 'feedback', 'Feedback']
    
    for col in text_columns:
        if col in df.columns:
            return col
    
    # Try to find any column with text data
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it contains text (average length > 10 chars)
            avg_length = df[col].dropna().astype(str).str.len().mean()
            if avg_length > 10:
                return col
    
    return None

def truncate_text(text, max_length=150):
    """Truncate text for display"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def predict_single_text(text, model, tokenizer):
    """Predict using the actual model for a single text"""
    try:
        if model is None:
            return {"success": False, "error": "Model not loaded"}
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Convert to sequence
        sequence = text_to_sequence(processed_text, tokenizer)
        
        if sequence is None:
            return {"success": False, "error": "Failed to convert text to sequence"}
        
        # Make prediction
        with torch.no_grad():
            target_output, severity_output = model(sequence)
            
            target_class = torch.argmax(target_output, dim=1).item()
            severity_class = torch.argmax(severity_output, dim=1).item()
            
            target_confidence = torch.max(torch.softmax(target_output, dim=1)).item()
            severity_confidence = torch.max(torch.softmax(severity_output, dim=1)).item()
        
        # Get class names
        target = TARGET_CLASSES.get(target_class, "Other")
        severity = SEVERITY_CLASSES.get(severity_class, "Neutral")
        
        confidence = (target_confidence + severity_confidence) / 2
        
        # Check if it's islamophobic
        is_islamophobic = "Yes" if severity in ["Promoting Violent", "Nonviolent_Hate"] and target in ["Islam", "Muslim"] else "No"
        
        return {
            "success": True,
            "predictions": {
                "target": target,
                "severity": severity,
                "islamophobia": is_islamophobic,
                "confidence": round(confidence, 3),
                "target_confidence": round(target_confidence, 3),
                "severity_confidence": round(severity_confidence, 3)
            }
        }
        
    except Exception as e:
        print(f"Model prediction error: {e}")
        return {"success": False, "error": str(e)}

def preprocess_text(text):
    """Preprocess text - match your training preprocessing"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

def text_to_sequence(text, tokenizer):
    """Convert text to sequence using tokenizer"""
    try:
        if tokenizer and hasattr(tokenizer, 'texts_to_sequences'):
            sequence = tokenizer.texts_to_sequences([text])[0]
        else:
            # Fallback: use simple word-to-index mapping
            words = text.split()
            vocab = get_fallback_vocab()
            sequence = []
            for word in words:
                if word in vocab:
                    sequence.append(vocab[word])
                else:
                    sequence.append(vocab["<UNK>"])
        
        # Pad sequence to MAX_LENGTH
        if len(sequence) > MAX_LENGTH:
            sequence = sequence[:MAX_LENGTH]
        else:
            sequence.extend([0] * (MAX_LENGTH - len(sequence)))
        
        return torch.tensor(sequence).unsqueeze(0)
        
    except Exception as e:
        print(f"Sequence conversion error: {e}")
        return None

def get_fallback_vocab():
    """Get fallback vocabulary for when tokenizer is not available"""
    vocab = {
        "<PAD>": 0, "<UNK>": 1,
        "islam": 2, "muslim": 3, "muslims": 4, "islamic": 5,
        "mosque": 6, "quran": 7, "allah": 8, "prophet": 9,
        "muhammad": 10, "sharia": 11, "hijab": 12, "halal": 13,
        "terror": 14, "terrorist": 15, "extremist": 16, "jihad": 17,
        "radical": 18, "violence": 19, "violent": 20, "attack": 21,
        "kill": 22, "murder": 23, "war": 24, "bomb": 25,
        "hate": 26, "hatred": 27, "racist": 28, "discrimination": 29,
        "peace": 30, "peaceful": 31, "tolerance": 32, "respect": 33,
        "community": 34, "diversity": 35, "love": 36, "understanding": 37,
        "anti": 38, "against": 39, "ban": 40, "stop": 41,
        "religion": 42, "religious": 43, "faith": 44, "worship": 45
    }
    return vocab

def rule_based_fallback(text):
    """Rule-based fallback classification when model fails"""
    text_lower = text.lower()
    
    # Target classification
    if any(word in text_lower for word in ['islam', 'quran', 'mosque', 'allah', 'prophet', 'sharia', 'islamic', 'muslim', 'muslims', 'islamist', 'islamists', 'jihad', 'jihadi', 'jihadist', 'shariah', 'halal', 'hijab', 'burqa', 'niqab', 
    'ramadan', 'eid', 'hajj', 'ummah', 'caliphate', 'fatwa', 'imam', 'mullah', 'ayatollah', 'islamic state', 'islamic law', 'islamic faith', 'islamic religion', 'islamic world', 'islamic culture']):
        target = "Islam"
    elif any(word in text_lower for word in ['muslim', 'muslims', 'muslim community', 'muslim people', 'moslem', 'moslems', 'muslim man', 'muslim woman', 'muslim girl', 'muslim boy', 'muslim family', 'muslim children',
    'muslim youth', 'muslim immigrant', 'muslim migrant', 'muslim refugee', 'muslim minority', 'muslim majority', 'muslim leader', 'muslim scholar', 'muslim cleric']):
        target = "Muslim"
    else:
        target = "Other"

    # Severity classification
    violent_words = ['kill', 'murder', 'attack', 'bomb', 'war', 'terror', 'violent', 'death', 'destroy', 'killing', 'assassinate', 'assassination', 'massacre', 'genocide', 'slaughter', 'exterminate', 'annihilate', 'execute', 
    'behead', 'decapitate', 'stone', 'flog', 'crucify', 'burn', 'shoot', 'stab', 'explode', 'bombing', 'terrorism', 'terrorist', 'terrorists', 'extremist', 'extremists', 'radical', 'radicals', 'jihad', 'jihadi', 'jihadist', 'holy war', 
    'crusade', 'crusader', 'vengeance', 'revenge', 'retaliate', 'retaliation', 'blood', 'bloodshed', 'carnage', 'violence']
    hate_words = ['hate', 'hatred', 'anti', 'against', 'ban', 'racist', 'discrimination', 'discriminate', 'prejudice', 'biased', 'bigot', 'bigotry', 'islamophobe', 'islamophobic', 'islamophobia', 'anti-muslim', 'anti-islam', 
    'xenophobe', 'xenophobic', 'xenophobia', 'intolerant', 'intolerance', 'despise', 'loathe', 'detest', 'abhor', 'disgust', 'disgusting', 'revolting', 'repulsive', 'contempt', 'scorn', 'deport', 'expel', 'exclude', 'segregate', 
    'separate', 'assimilate', 'purge', 'cleanse', 'wipe out', 'eliminate', 'eradicate', 'remove', 'get rid of', 'go back', 'go home', 'not welcome', 'unwelcome', 'invader', 'invasion', 'takeover', 'conquest', 'flood', 'swarm', 'overrun']
    peaceful_words = ['peace', 'peaceful', 'love', 'respect', 'tolerance', 'community', 'harmony', 'coexist', 'coexistence', 'understanding', 'compassion', 'kindness', 'empathy', 'dialogue', 'discussion', 'conversation', 
    'friendship', 'friends', 'brotherhood', 'sisterhood', 'solidarity', 'unity', 'together', 'hope', 'hopeful', 'optimism', 'joy', 'happiness', 'gratitude', 'thanks', 'thankful', 'diversity', 'inclusive', 'inclusion', 'multicultural', 
    'pluralism', 'equal', 'equality', 'fair', 'fairness', 'rights', 'human rights', 'religious freedom', 'education', 'learn', 'understanding', 'awareness', 'knowledge', 'wisdom', 'enlightenment', 'help', 'assist', 'support', 
    'encourage', 'cooperate', 'collaborate', 'build', 'develop', 'improve', 'progress', 'faith', 'belief', 'spiritual', 'prayer', 'worship', 'devotion', 'pious', 'religious', 'god', 'blessing', 'bless', 'grace', 'sacred', 'holy', 
    'moderate', 'moderation', 'balanced', 'reasonable', 'rational', 'logical', 'sensible', 'pragmatic', 'diplomatic', 'negotiate', 'solution', 'resolve', 'agreement', 'consensus', 'success', 'achieve', 'benefit', 'positive']
    
    if any(word in text_lower for word in violent_words):
        severity = "Promoting Violent"
        confidence = 0.85
    elif any(word in text_lower for word in hate_words):
        severity = "Nonviolent_Hate"
        confidence = 0.75
    elif any(word in text_lower for word in peaceful_words):
        severity = "Neutral"
        confidence = 0.90
    else:
        severity = "Neutral"
        confidence = 0.80
    
    return target, severity, confidence

def calculate_statistics(results):
    """Calculate statistics from results"""
    if not results:
        return {
            "total_records": 0,
            "islamophobia_detected": 0,
            "target_distribution": {"Islam": 0, "Muslim": 0, "Other": 0},
            "severity_distribution": {"Promoting Violent": 0, "Nonviolent_Hate": 0, "Neutral": 0}
        }
    
    target_counts = {"Islam": 0, "Muslim": 0, "Other": 0}
    severity_counts = {"Promoting Violent": 0, "Nonviolent_Hate": 0, "Neutral": 0}
    islamophobia_count = 0
    
    for result in results:
        target_counts[result["target"]] += 1
        severity_counts[result["severity"]] += 1
        if result["islamophobia"] == "Yes":
            islamophobia_count += 1
    
    total = len(results)
    
    return {
        "total_records": total,
        "islamophobia_detected": islamophobia_count,
        "target_distribution": {
            "Islam": round((target_counts["Islam"] / total) * 100, 1),
            "Muslim": round((target_counts["Muslim"] / total) * 100, 1),
            "Other": round((target_counts["Other"] / total) * 100, 1)
        },
        "severity_distribution": {
            "Promoting Violent": round((severity_counts["Promoting Violent"] / total) * 100, 1),
            "Nonviolent_Hate": round((severity_counts["Nonviolent_Hate"] / total) * 100, 1),
            "Neutral": round((severity_counts["Neutral"] / total) * 100, 1)
        }
    }

# ===== TEST FUNCTION =====
if __name__ == "__main__":
    print("BatchApp.py - Test Mode")
    print("This module provides batch processing functions for TextApp.py")
    print("To test, run: python TextApp.py")