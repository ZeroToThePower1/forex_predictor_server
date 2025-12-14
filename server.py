import os
import json
import sqlite3
import uvicorn
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dotenv import load_dotenv

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
# DATABASE SETUP (KEEP THIS THE SAME)
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
# MODEL INPUT FORMATS (KEEP THIS THE SAME)
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
# HELPER FUNCTIONS - UPDATED FOR XGBOOST 2.0
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
    
    # Load new models - USING xgb.Booster() instead of XGBClassifier()
    pair_events = []
    for file in os.listdir(pair_dir):
        if file.endswith(".json") and not file.endswith("_accuracies.json"):
            model_base_name = file.replace(".json", "")
            model_full_name = f"{pair_name}_{model_base_name}"
            
            try:
                # CHANGED: Use xgb.Booster() for XGBoost 2.0
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

def predict_with_model(model, features):
    """Make prediction with xgb.Booster model (XGBoost 2.0)"""
    # Convert to DMatrix (required by xgb.Booster)
    dmatrix = xgb.DMatrix(np.array([features]))
    
    # Get prediction probability
    prediction = model.predict(dmatrix)[0]
    
    # XGBoost returns probability for class 1
    prob = float(prediction)
    
    # Convert to binary prediction (0 or 1)
    pred_class = 1 if prob >= 0.5 else 0
    
    return pred_class, prob

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

# ==================================================
# FASTAPI APP WITH CORS (KEEP THIS THE SAME)
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================================
# ROUTES - UPDATED FOR XGBOOST 2.0
# ==================================================

# ---------- PAIR MANAGEMENT (KEEP SAME) ----------
@app.get("/pairs")
def get_available_pairs():
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
def switch_pair(request: SwitchPairRequest):
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
        raise HTTPException(status_code=500, detail=f"Failed to switch pair: {str(e)}")

@app.get("/pairs/{pair_name}/models")
def get_pair_models(pair_name: str):
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

# ---------- EVENT PREDICTION - UPDATED ----------
@app.post("/predict/event")
def predict_event(data: EventInput):
    """Predict single event direction"""
    # Try different naming patterns
    possible_names = [
        data.event,  # Exact match
        f"{current_pair}_{data.event}",  # With current pair prefix
        data.event.replace(f"{current_pair}_", "")  # Without pair prefix if present
    ]
    
    model_to_use = None
    for name in possible_names:
        if name in event_models:
            model_to_use = name
            break
    
    if not model_to_use:
        # Try to find any model containing the event name
        matching = [m for m in event_models.keys() if data.event in m]
        if matching:
            model_to_use = matching[0]
        else:
            raise HTTPException(status_code=404, detail=f"Event model '{data.event}' not found")
    
    model = event_models[model_to_use]
    
    # Make prediction - USING UPDATED FUNCTION
    features = [data.actual, data.forecast, data.previous]
    pred, prob = predict_with_model(model, features)
    
    # Calculate derived values
    prediction_label = "UP" if pred == 1 else "DOWN"
    confidence_percent = round(prob * 100, 2)
    model_accuracy = model_accuracies.get(model_to_use, 0.5)
    
    # Determine signal strength
    if prob > 0.8:
        signal_strength = "STRONG"
    elif prob > 0.7:
        signal_strength = "MODERATE"
    elif prob > 0.6:
        signal_strength = "WEAK"
    else:
        signal_strength = "VERY WEAK"
    
    signal = f"{signal_strength} {'BUY' if pred == 1 else 'SELL'}"
    
    # Prepare response
    response = {
        "event": data.event,
        "full_model_name": model_to_use,
        "pair_name": current_pair,
        "prediction": int(pred),
        "prediction_label": prediction_label,
        "confidence": prob,
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
    
    # Save to database
    prediction_id = save_prediction_to_db("event", response)
    response["prediction_id"] = prediction_id
    
    return response

# ---------- ENSEMBLE PREDICTION (KEEP SAME) ----------
# ... (KEEP YOUR EXISTING ENSEMBLE CODE, IT SHOULD WORK) ...

# ==================================================
# REST OF YOUR ROUTES (KEEP THEM AS THEY ARE)
# ==================================================
# ... (COPY ALL YOUR OTHER ROUTES FROM BEFORE - THEY DON'T NEED TO CHANGE) ...

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
        reload=True,
        log_level="info"
    )