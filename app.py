import os
from flask import Flask, render_template, request, jsonify
from sentiment_engine import HybridSentimentEngine
from dataset_loader import DatasetLoader

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(ROOT_DIR, 'templates')
STATIC_DIR = os.path.join(ROOT_DIR, 'static')

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

DATASET_PATH = os.path.join(ROOT_DIR, 'twitter_training.csv')
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.abspath(os.path.join(ROOT_DIR, '..', 'twitter_training.csv'))

engine = HybridSentimentEngine()
loader = DatasetLoader(DATASET_PATH)

_initialized = False

def ensure_initialized():
    global _initialized
    if not _initialized:
        try:
            if os.path.exists(DATASET_PATH):
                loader.load(max_records=10000)
                samples = loader.get_training_samples(sample_size=1000)
                engine.train_ml_model(samples, top_n_features=300)
        except Exception as e:
            print(f"Lazy initialization notice: {e}")
        _initialized = True

@app.route('/')
@app.route('/api/index')
def index():
    ensure_initialized()
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    ensure_initialized()
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = engine.analyze(text)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    ensure_initialized()
    summary = loader.get_summary()
    return jsonify(summary)

@app.route('/api/entity/<name>', methods=['GET'])
def get_entity_details(name):
    ensure_initialized()
    details = loader.get_entity_breakdown(name)
    return jsonify(details)

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    ensure_initialized()
    data = request.get_json() or {}
    texts = data.get('texts', [])
    
    if not isinstance(texts, list):
        return jsonify({'error': 'Invalid format. Expected array of texts.'}), 400

    results = [engine.analyze(t) for t in texts[:50]]
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
