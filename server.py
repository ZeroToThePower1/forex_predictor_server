import os
import json
import sqlite3
import uvicorn
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import traceback

# ==================================================
# LOAD ENVIRONMENT VARIABLES - DO THIS FIRST!
# ==================================================
load_dotenv()

# ==================================================
# CONFIGURATION FROM ENVIRONMENT VARIABLES
# ==================================================
MODELS_BASE_DIR = os.getenv("MODELS_BASE_DIR", "models")
DEFAULT_PAIR = os.getenv("DEFAULT_PAIR", "EUR_USD")
DATABASE_PATH = os.getenv("DATABASE_PATH", "predictions.db")
PORT = int(os.getenv("PORT", 8000))

# ==================================================
# GLOBAL STATE
# ==================================================
current_pair = DEFAULT_PAIR
event_models = {}  # Will store xgb.Booster objects
available_events = []
model_accuracies = {}
pair_models = {}
available_pairs = []

# ==================================================
# DATABASE SETUP
# ==================================================
def init_database():
    """Initialize SQLite database for storing predictions and history"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Create predictions history table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_type TEXT NOT NULL,
        pair_name TEXT NOT NULL,
        event_name TEXT,
        prediction INTEGER,
        prediction_label TEXT,
        confidence REAL,
        signal TEXT,
        inputs_json TEXT,
        result_json TEXT,
        actual_result REAL,
        actual_label TEXT,
        is_correct INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create pair configurations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pair_configs (
        pair_name TEXT PRIMARY KEY,
        display_name TEXT,
        base_currency TEXT,
        quote_currency TEXT,
        yf_symbol TEXT,
        is_active INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create ensemble predictions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS ensemble_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pair_name TEXT NOT NULL,
        prediction INTEGER,
        prediction_label TEXT,
        confidence REAL,
        up_probability REAL,
        down_probability REAL,
        signal TEXT,
        method TEXT,
        event_count INTEGER,
        events_json TEXT,
        calculation_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DATABASE_PATH}")

# ==================================================
# MODEL INPUT FORMATS
# ==================================================
class EventInput(BaseModel):
    event: str
    actual: float
    forecast: float
    previous: float

class EnsembleRequest(BaseModel):
    events: List[EventInput]
    method: str = "average"  # "average", "weighted", "confident"
    pair_name: Optional[str] = None

class UpdateResultRequest(BaseModel):
    prediction_id: int
    actual_price_change: Optional[float] = None
    actual_direction: Optional[str] = None  # "UP" or "DOWN"

class SwitchPairRequest(BaseModel):
    pair_name: str

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def get_db_connection():
    """Get SQLite database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def discover_available_pairs():
    """Discover all available currency pairs from models directory"""
    pairs = []
    if os.path.exists(MODELS_BASE_DIR):
        for item in os.listdir(MODELS_BASE_DIR):
            item_path = os.path.join(MODELS_BASE_DIR, item)
            if os.path.isdir(item_path) and not item.startswith('.'):
                # Check if directory contains model files
                model_files = [f for f in os.listdir(item_path) if f.endswith('.json')]
                if model_files:
                    pairs.append(item)
    
    # Update database with discovered pairs
    conn = get_db_connection()
    cursor = conn.cursor()
    for pair in pairs:
        cursor.execute('''
            INSERT OR IGNORE INTO pair_configs (pair_name, display_name, is_active)
            VALUES (?, ?, ?)
        ''', (pair, pair.replace('_', '/'), 1 if pair == DEFAULT_PAIR else 0))
    conn.commit()
    conn.close()
    
    return sorted(pairs)

def load_models_for_pair(pair_name: str):
    """Load all models for a specific currency pair"""
    global event_models, available_events, model_accuracies, pair_models, current_pair
    
    pair_dir = os.path.join(MODELS_BASE_DIR, pair_name)
    if not os.path.exists(pair_dir):
        raise HTTPException(status_code=404, detail=f"Models directory for pair '{pair_name}' not found")
    
    print(f"\n🔍 Loading models for pair: {pair_name}")
    
    # Clear existing models for this pair
    models_to_remove = [m for m in event_models.keys() if m.startswith(f"{pair_name}_")]
    for model in models_to_remove:
        del event_models[model]
    
    # Load new models
    pair_events = []
    for file in os.listdir(pair_dir):
        if file.endswith(".json") and not file.endswith("_accuracies.json"):
            model_base_name = file.replace(".json", "")
            model_full_name = f"{pair_name}_{model_base_name}"
            
            try:
                model = xgb.Booster()
                model.load_model(os.path.join(pair_dir, file))
                event_models[model_full_name] = model
                pair_events.append(model_full_name)
                print(f"  ✔ Loaded: {model_base_name}")
            except Exception as e:
                print(f"  ❌ Failed to load {file}: {e}")
    
    # Load accuracies if available
    acc_path = os.path.join(pair_dir, "model_accuracies.json")
    if os.path.exists(acc_path):
        try:
            with open(acc_path, 'r') as f:
                pair_accuracies = json.load(f)
                for model_name, acc in pair_accuracies.items():
                    full_name = f"{pair_name}_{model_name}"
                    model_accuracies[full_name] = acc
            print(f"  📊 Loaded accuracies for {pair_name}")
        except Exception as e:
            print(f"  ⚠ Could not load accuracies: {e}")
    
    # Update global state
    pair_models[pair_name] = pair_events
    available_events = list(event_models.keys())
    current_pair = pair_name
    
    print(f"✅ Loaded {len(pair_events)} models for {pair_name}")
    
    # Update database
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE pair_configs SET is_active = 0')
    cursor.execute('UPDATE pair_configs SET is_active = 1 WHERE pair_name = ?', (pair_name,))
    conn.commit()
    conn.close()
    
    return {
        "pair": pair_name,
        "models_loaded": len(pair_events),
        "total_models": len(event_models)
    }

def predict_with_model(model, features, feature_names=None):
    """Make prediction with xgb.Booster model"""
    # Try with cleaned feature names (what your models expect)
    if feature_names is None:
        feature_names = ['actual_clean', 'forecast_clean', 'previous_clean']
    
    try:
        dmatrix = xgb.DMatrix(
            np.array([features]),
            feature_names=feature_names
        )
        
        prediction = model.predict(dmatrix)[0]
        prob = float(prediction)
        pred_class = 1 if prob >= 0.5 else 0
        
        return pred_class, prob
        
    except Exception as e:
        # If cleaned names don't work, try without feature names
        print(f"⚠ Warning: Feature name error, trying without names: {e}")
        try:
            dmatrix = xgb.DMatrix(np.array([features]))
            prediction = model.predict(dmatrix)[0]
            prob = float(prediction)
            pred_class = 1 if prob >= 0.5 else 0
            return pred_class, prob
        except Exception as e2:
            raise Exception(f"Prediction failed: {e2}")

def save_prediction_to_db(prediction_type: str, data: dict):
    """Save prediction to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (
            prediction_type, pair_name, event_name, prediction, prediction_label,
            confidence, signal, inputs_json, result_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_type,
        data.get('pair_name', current_pair),
        data.get('event'),
        data.get('prediction'),
        data.get('prediction_label'),
        data.get('confidence'),
        data.get('signal'),
        json.dumps(data.get('inputs', {})),
        json.dumps(data),
        datetime.now().isoformat()
    ))
    
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return prediction_id

def save_ensemble_to_db(data: dict):
    """Save ensemble prediction to database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO ensemble_predictions (
            pair_name, prediction, prediction_label, confidence, up_probability,
            down_probability, signal, method, event_count, events_json, calculation_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        data.get('pair_name', current_pair),
        data.get('prediction'),
        data.get('prediction_label'),
        data.get('confidence'),
        data.get('up_probability', 0),
        data.get('down_probability', 0),
        data.get('signal'),
        data.get('method'),
        data.get('event_count', 0),
        json.dumps(data.get('individual_predictions', [])),
        json.dumps(data.get('calculation_steps', []))
    ))
    
    ensemble_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return ensemble_id

def get_prediction_history(prediction_type: str = None, limit: int = 50):
    """Get prediction history from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = '''
        SELECT id, prediction_type, pair_name, event_name, prediction, 
               prediction_label, confidence, signal, created_at
        FROM predictions
    '''
    params = []
    
    if prediction_type:
        query += ' WHERE prediction_type = ?'
        params.append(prediction_type)
    
    query += ' ORDER BY created_at DESC LIMIT ?'
    params.append(limit)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    predictions = []
    for row in rows:
        predictions.append({
            "id": row["id"],
            "prediction_type": row["prediction_type"],
            "pair_name": row["pair_name"],
            "event_name": row["event_name"],
            "prediction": row["prediction"],
            "prediction_label": row["prediction_label"],
            "confidence": row["confidence"],
            "signal": row["signal"],
            "created_at": row["created_at"]
        })
    
    return predictions

def get_ensemble_history(limit: int = 50):
    """Get ensemble prediction history from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, pair_name, prediction, prediction_label, confidence, 
               up_probability, down_probability, signal, method, event_count, created_at
        FROM ensemble_predictions
        ORDER BY created_at DESC LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    ensembles = []
    for row in rows:
        ensembles.append({
            "id": row["id"],
            "pair_name": row["pair_name"],
            "prediction": row["prediction"],
            "prediction_label": row["prediction_label"],
            "confidence": row["confidence"],
            "up_probability": row["up_probability"],
            "down_probability": row["down_probability"],
            "signal": row["signal"],
            "method": row["method"],
            "event_count": row["event_count"],
            "created_at": row["created_at"]
        })
    
    return ensembles

def get_performance_stats():
    """Get performance statistics from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute('SELECT COUNT(*) as total FROM predictions')
    total = cursor.fetchone()["total"]
    
    # Total correct predictions (where actual_result is set and is_correct = 1)
    cursor.execute('SELECT COUNT(*) as correct FROM predictions WHERE is_correct = 1')
    correct = cursor.fetchone()["correct"]
    
    # Accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Distribution
    cursor.execute('SELECT prediction_label, COUNT(*) as count FROM predictions GROUP BY prediction_label')
    distribution = {row["prediction_label"]: row["count"] for row in cursor.fetchall()}
    
    conn.close()
    
    return {
        "total_predictions": total,
        "correct_predictions": correct,
        "accuracy": accuracy,
        "distribution": distribution
    }

# ==================================================
# FASTAPI APP
# ==================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Forex Predictor API...")
    init_database()
    
    # Discover available pairs
    global available_pairs
    available_pairs = discover_available_pairs()
    print(f"📊 Discovered {len(available_pairs)} currency pairs: {available_pairs}")
    
    # Load default pair
    if available_pairs:
        try:
            load_models_for_pair(DEFAULT_PAIR)
        except Exception as e:
            print(f"⚠ Failed to load default pair {DEFAULT_PAIR}: {e}")
            if available_pairs:
                load_models_for_pair(available_pairs[0])
    else:
        print("⚠ No currency pairs found in models directory!")
    
    print(f"📈 Current pair: {current_pair}")
    print(f"📊 Total models loaded: {len(event_models)}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Forex Predictor API...")

app = FastAPI(
    title="Forex Predictor API",
    version="2.0",
    description="Advanced Forex Prediction System with Multi-Pair Support",
    lifespan=lifespan
)

# ==================================================
# CORS MIDDLEWARE - FIXED FOR GITHUB PAGES
# ==================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://zerotothepower1.github.io",  # Your GitHub Pages
        "http://localhost:8000",
        "http://127.0.0.1:8000", 
        "http://localhost:3000",
        "http://localhost:5500",
        "https://forex-predictor-server.onrender.com",
        "*"  # Fallback for testing
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=600,
)

# ==================================================
# ROUTES
# ==================================================

@app.get("/")
async def root():
    """Root endpoint - API status"""
    return {
        "message": "Forex Predictor API",
        "version": "2.0",
        "status": "online",
        "current_pair": current_pair,
        "models_loaded": len(event_models),
        "available_pairs": len(available_pairs),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/pairs")
async def get_available_pairs():
    """Get list of all available currency pairs"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM pair_configs ORDER BY pair_name')
    pairs = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return {
        "available_pairs": available_pairs,
        "current_pair": current_pair,
        "pair_details": pairs,
        "total_pairs": len(available_pairs)
    }

@app.post("/pairs/switch")
async def switch_pair(request: SwitchPairRequest):
    """Switch to a different currency pair"""
    if request.pair_name not in available_pairs:
        raise HTTPException(status_code=404, detail=f"Pair '{request.pair_name}' not found")
    
    try:
        result = load_models_for_pair(request.pair_name)
        return {
            "message": f"Successfully switched to {request.pair_name}",
            "current_pair": current_pair,
            "details": result
        }
    except Exception as e:
        print(f"❌ Error switching pair: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to switch pair: {str(e)}")

@app.get("/pairs/{pair_name}/models")
async def get_pair_models(pair_name: str):
    """Get all models available for a specific pair"""
    if pair_name not in available_pairs:
        raise HTTPException(status_code=404, detail=f"Pair '{pair_name}' not found")
    
    pair_dir = os.path.join(MODELS_BASE_DIR, pair_name)
    models = []
    
    if os.path.exists(pair_dir):
        for file in os.listdir(pair_dir):
            if file.endswith(".json") and not file.endswith("_accuracies.json"):
                model_name = file.replace(".json", "")
                models.append({
                    "name": model_name,
                    "full_name": f"{pair_name}_{model_name}",
                    "accuracy": model_accuracies.get(f"{pair_name}_{model_name}", 0.5)
                })
    
    return {
        "pair": pair_name,
        "models": sorted(models, key=lambda x: x["name"]),
        "count": len(models)
    }

@app.get("/events")
async def get_events():
    """Get all available events for current pair"""
    models = []
    for model_name in pair_models.get(current_pair, []):
        base_name = model_name.replace(f"{current_pair}_", "")
        models.append({
            "name": base_name,
            "full_name": model_name,
            "accuracy": model_accuracies.get(model_name, 0.5)
        })
    
    return {
        "pair": current_pair,
        "events": sorted(models, key=lambda x: x["name"]),
        "count": len(models)
    }

@app.post("/predict/event")
async def predict_event(data: EventInput):
    """Predict single event direction"""
    try:
        print(f"\n🎯 PREDICTION REQUEST RECEIVED")
        print(f"📊 Event: {data.event}")
        print(f"📈 Inputs: actual={data.actual}, forecast={data.forecast}, previous={data.previous}")
        print(f"📍 Current pair: {current_pair}")
        print(f"📦 Total models loaded: {len(event_models)}")
        
        # Try different naming patterns
        possible_names = [
            data.event,
            f"{current_pair}_{data.event}",
            data.event.replace(f"{current_pair}_", "")
        ]
        
        print(f"🔍 Looking for model with names: {possible_names}")
        
        model_to_use = None
        for name in possible_names:
            if name in event_models:
                model_to_use = name
                print(f"✅ Found exact match: {model_to_use}")
                break
        
        if not model_to_use:
            # Try to find any model containing the event name
            matching = [m for m in event_models.keys() if data.event in m]
            print(f"🔍 Fuzzy matching results: {matching}")
            if matching:
                model_to_use = matching[0]
                print(f"✅ Fuzzy matched model: {model_to_use}")
            else:
                error_msg = f"Event model '{data.event}' not found."
                print(f"❌ {error_msg}")
                print(f"📋 Available models: {list(event_models.keys())[:10]}...")  # First 10
                raise HTTPException(status_code=404, detail=error_msg)
        
        model = event_models[model_to_use]
        features = [data.actual, data.forecast, data.previous]
        print(f"🔢 Features array: {features}")
        
        # Make prediction - TRY WITH CLEANED FEATURE NAMES FIRST
        try:
            pred, prob = predict_with_model(
                model, 
                features,
                feature_names=['actual_clean', 'forecast_clean', 'previous_clean']
            )
            print(f"✅ Prediction successful with cleaned feature names")
        except Exception as e:
            print(f"⚠ Cleaned names failed, trying without feature names: {e}")
            pred, prob = predict_with_model(model, features, feature_names=None)
        
        print(f"🎯 Prediction result: {pred} ({'UP' if pred == 1 else 'DOWN'}), Probability: {prob:.3f}")
        
        prediction_label = "UP" if pred == 1 else "DOWN"
        confidence_percent = round(prob * 100, 2)
        model_accuracy = model_accuracies.get(model_to_use, 0.5)
        
        if prob > 0.8:
            signal_strength = "STRONG"
        elif prob > 0.7:
            signal_strength = "MODERATE"
        elif prob > 0.6:
            signal_strength = "WEAK"
        else:
            signal_strength = "VERY WEAK"
        
        signal = f"{signal_strength} {'BUY' if pred == 1 else 'SELL'}"
        
        response = {
            "event": data.event,
            "full_model_name": model_to_use,
            "pair_name": current_pair,
            "prediction": int(pred),
            "prediction_label": prediction_label,
            "confidence": float(prob),
            "confidence_percent": confidence_percent,
            "model_accuracy": model_accuracy,
            "signal": signal,
            "signal_strength": signal_strength,
            "inputs": {
                "actual": data.actual,
                "forecast": data.forecast,
                "previous": data.previous
            },
            "timestamp": datetime.now().isoformat()
        }
        
        prediction_id = save_prediction_to_db("event", response)
        response["prediction_id"] = prediction_id
        
        print(f"✅ Prediction saved to DB with ID: {prediction_id}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"🔥 CRITICAL ERROR in predict_event: {str(e)}")
        print(f"🔥 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/predict/ensemble")
async def predict_ensemble(request: EnsembleRequest):
    """Make ensemble prediction from multiple events"""
    pair_name = request.pair_name or current_pair
    
    individual_results = []
    calculation_steps = []
    
    print(f"\n🎯 ENSEMBLE PREDICTION REQUEST")
    print(f"📊 Method: {request.method}")
    print(f"📍 Pair: {pair_name}")
    print(f"📈 Number of events: {len(request.events)}")
    
    # Get predictions for each event
    for event_input in request.events:
        try:
            # Try different naming patterns
            possible_names = [
                event_input.event,
                f"{pair_name}_{event_input.event}",
                event_input.event.replace(f"{pair_name}_", "")
            ]
            
            model_to_use = None
            for name in possible_names:
                if name in event_models:
                    model_to_use = name
                    break
            
            if not model_to_use:
                matching = [m for m in event_models.keys() if event_input.event in m]
                if matching:
                    model_to_use = matching[0]
                else:
                    print(f"⚠ Skipping event {event_input.event}: model not found")
                    continue  # Skip events without models
            
            model = event_models[model_to_use]
            features = [event_input.actual, event_input.forecast, event_input.previous]
            
            # Make prediction
            try:
                pred, prob = predict_with_model(
                    model, 
                    features,
                    feature_names=['actual_clean', 'forecast_clean', 'previous_clean']
                )
            except:
                pred, prob = predict_with_model(model, features, feature_names=None)
            
            # Convert DOWN predictions to UP probabilities for consistent calculation
            up_probability = prob if pred == 1 else (1 - prob)
            
            individual_results.append({
                "event": event_input.event,
                "prediction": int(pred),
                "prediction_label": "UP" if pred == 1 else "DOWN",
                "probability": prob,
                "up_probability": up_probability,
                "model_accuracy": model_accuracies.get(model_to_use, 0.5)
            })
            
            calculation_steps.append({
                "event": event_input.event,
                "original_prediction": "UP" if pred == 1 else "DOWN",
                "original_probability": prob,
                "up_probability": up_probability
            })
            
            print(f"  ✅ {event_input.event}: {'UP' if pred == 1 else 'DOWN'} ({prob:.3f})")
            
        except Exception as e:
            print(f"  ❌ Error predicting event {event_input.event}: {e}")
            continue
    
    if not individual_results:
        raise HTTPException(status_code=400, detail="No valid event predictions")
    
    print(f"📊 Valid predictions: {len(individual_results)}/{len(request.events)}")
    
    # Apply ensemble method
    if request.method == "weighted":
        # Weight by model accuracy
        total_weight = sum(r["model_accuracy"] for r in individual_results)
        up_probability = sum(r["up_probability"] * r["model_accuracy"] for r in individual_results) / total_weight
        method_used = "weighted_average"
    elif request.method == "confident":
        # Use only high-confidence predictions
        confident_results = [r for r in individual_results if r["probability"] > 0.7 or r["probability"] < 0.3]
        if confident_results:
            up_probability = sum(r["up_probability"] for r in confident_results) / len(confident_results)
            method_used = f"confident_voting ({len(confident_results)} events)"
        else:
            up_probability = sum(r["up_probability"] for r in individual_results) / len(individual_results)
            method_used = "average (no confident events)"
    else:  # average
        up_probability = sum(r["up_probability"] for r in individual_results) / len(individual_results)
        method_used = "simple_average"
    
    down_probability = 1 - up_probability
    final_prediction = 1 if up_probability >= 0.5 else 0
    final_confidence = up_probability if final_prediction == 1 else down_probability
    
    # Determine signal strength
    if final_confidence > 0.8:
        signal_strength = "STRONG"
    elif final_confidence > 0.7:
        signal_strength = "MODERATE"
    elif final_confidence > 0.6:
        signal_strength = "WEAK"
    else:
        signal_strength = "VERY WEAK"
    
    signal = f"{signal_strength} {'BUY' if final_prediction == 1 else 'SELL'}"
    
    response = {
        "pair_name": pair_name,
        "prediction": int(final_prediction),
        "prediction_label": "UP" if final_prediction == 1 else "DOWN",
        "confidence": final_confidence,
        "confidence_percent": round(final_confidence * 100, 2),
        "up_probability": up_probability,
        "down_probability": down_probability,
        "signal": signal,
        "signal_strength": signal_strength,
        "method": method_used,
        "individual_predictions": individual_results,
        "calculation_steps": calculation_steps,
        "event_count": len(individual_results),
        "timestamp": datetime.now().isoformat()
    }
    
    ensemble_id = save_ensemble_to_db(response)
    response["ensemble_id"] = ensemble_id
    
    print(f"🎯 Ensemble result: {response['prediction_label']} ({final_confidence:.3f})")
    print(f"📊 Method: {method_used}")
    
    return response

@app.get("/predictions/history")
async def get_prediction_history_endpoint(
    type: Optional[str] = Query(None, description="Type: 'event' or 'ensemble'"),
    limit: int = Query(50, description="Number of records to return")
):
    """Get prediction history"""
    try:
        if type == "event":
            predictions = get_prediction_history("event", limit)
            return {"predictions": predictions}
        elif type == "ensemble":
            ensembles = get_ensemble_history(limit)
            return {"predictions": ensembles}
        else:
            # Return recent predictions (mixed)
            events = get_prediction_history(None, limit)
            return {"predictions": events}
    except Exception as e:
        print(f"❌ Error getting history: {e}")
        return {"predictions": []}

@app.get("/performance/stats")
async def get_performance_stats_endpoint():
    """Get performance statistics"""
    try:
        stats = get_performance_stats()
        return stats
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return {
            "total_predictions": 0,
            "correct_predictions": 0,
            "accuracy": 0,
            "distribution": {}
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "current_pair": current_pair,
        "models_loaded": len(event_models),
        "database": "connected" if os.path.exists(DATABASE_PATH) else "missing",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/docs")
async def custom_docs_redirect():
    """Redirect to Swagger docs"""
    return JSONResponse(content={"docs_url": "/docs"})

# ==================================================
# DEBUG ENDPOINTS
# ==================================================
@app.get("/debug/models")
async def debug_models():
    """Debug endpoint to see loaded models"""
    return {
        "current_pair": current_pair,
        "total_models": len(event_models),
        "models_by_pair": {pair: len(models) for pair, models in pair_models.items()},
        "available_pairs": available_pairs,
        "sample_models": list(event_models.keys())[:5] if event_models else []
    }

@app.get("/debug/test-prediction")
async def debug_test_prediction():
    """Test prediction with dummy data"""
    test_data = EventInput(
        event="GDP_Growth",  # Change to your actual event name
        actual=1.5,
        forecast=1.2,
        previous=1.3
    )
    
    try:
        # Call the actual prediction function
        result = await predict_event(test_data)
        return {"status": "success", "result": result}
    except Exception as e:
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc()}

# ==================================================
# MAIN ENTRY POINT
# ==================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("FOREX PREDICTOR API v2.0 - STARTING SERVER")
    print("="*60)
    print(f"📡 Server starting on http://0.0.0.0:{PORT}")
    print(f"📊 API Documentation: http://0.0.0.0:{PORT}/docs")
    print(f"📈 Models directory: {MODELS_BASE_DIR}")
    print(f"📁 Default pair: {DEFAULT_PAIR}")
    print(f"💾 Database: {DATABASE_PATH}")
    print("="*60)
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=PORT,
        reload=False,
        log_level="info"
    )
