import os
from flask import Flask, render_template, request, jsonify
from sentiment_engine import HybridSentimentEngine
from dataset_loader import DatasetLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Dataset path resolution (checking root or parent dir)
DATASET_PATH = os.path.join(ROOT_DIR, 'twitter_training.csv')
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.abspath(os.path.join(ROOT_DIR, '..', 'twitter_training.csv'))

engine = HybridSentimentEngine()
loader = DatasetLoader(DATASET_PATH)

_initialized = False

def ensure_initialized():
    """Idempotent initialization for model training and dataset preloading."""
    global _initialized
    if not _initialized:
        try:
            if os.path.exists(DATASET_PATH):
                max_records = int(os.environ.get('MAX_DATASET_RECORDS', 10000))
                sample_size = int(os.environ.get('ML_SAMPLE_SIZE', 1500))
                top_features = int(os.environ.get('ML_TOP_FEATURES', 400))
                
                loader.load(max_records=max_records)
                samples = loader.get_training_samples(sample_size=sample_size)
                engine.train_ml_model(samples, top_n_features=top_features)
        except Exception as e:
            print(f"Dataset initialization notice: {e}")
        _initialized = True

@app.route('/')
@app.route('/api/index')
def index():
    ensure_initialized()
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    ensure_initialized()
    return jsonify({
        'status': 'healthy',
        'dataset_loaded': loader.loaded,
        'records_count': len(loader.data),
        'ml_trained': engine._trained,
        'features_count': len(engine.word_features)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ensure_initialized()
    if not request.is_json:
        return jsonify({'error': 'Invalid Content-Type. Expected application/json.'}), 400

    data = request.get_json() or {}
    text = data.get('text', '')
    
    if not isinstance(text, str) or not text.strip():
        return jsonify({'error': 'No valid text provided for analysis.'}), 400

    result = engine.analyze(text.strip())
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    ensure_initialized()
    summary = loader.get_summary(engine=engine)
    return jsonify(summary)

@app.route('/api/entity/<name>', methods=['GET'])
def get_entity_details(name):
    ensure_initialized()
    if not name or not isinstance(name, str):
        return jsonify({'error': 'Entity name is required.'}), 400
        
    details = loader.get_entity_breakdown(name.strip())
    return jsonify(details)

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    ensure_initialized()
    if not request.is_json:
        return jsonify({'error': 'Invalid Content-Type. Expected application/json.'}), 400

    data = request.get_json() or {}
    texts = data.get('texts', [])
    
    if not isinstance(texts, list):
        return jsonify({'error': 'Invalid format. Expected array of texts.'}), 400

    # Filter out non-string items and limit batch size to 50
    valid_texts = [str(t).strip() for t in texts if t is not None and str(t).strip()][:50]
    
    if not valid_texts:
        return jsonify({'error': 'No non-empty text items found in batch.'}), 400

    results = [engine.analyze(t) for t in valid_texts]
    return jsonify({
        'total_processed': len(results),
        'results': results
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() in ['true', '1']
    app.run(host='0.0.0.0', port=port, debug=debug)
