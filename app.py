import os
from flask import Flask, render_template, request, jsonify
from sentiment_engine import HybridSentimentEngine
from dataset_loader import DatasetLoader

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, 'twitter_training.csv')
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = os.path.abspath(os.path.join(BASE_DIR, '..', 'twitter_training.csv'))

# Initialize engine & loader
engine = HybridSentimentEngine()
loader = DatasetLoader(DATASET_PATH)

print("Loading dataset...")
total = loader.load()
print(f"Loaded {total} dataset records.")

print("Training Naive Bayes ML model on dataset samples...")
training_samples = loader.get_training_samples(sample_size=3000)
engine.train_ml_model(training_samples)
print("ML Model trained successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    result = engine.analyze(text)
    return jsonify(result)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    summary = loader.get_summary()
    return jsonify(summary)

@app.route('/api/entity/<name>', methods=['GET'])
def get_entity_details(name):
    details = loader.get_entity_breakdown(name)
    return jsonify(details)

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    data = request.get_json() or {}
    texts = data.get('texts', [])
    
    if not isinstance(texts, list):
        return jsonify({'error': 'Invalid format. Expected array of texts.'}), 400

    results = [engine.analyze(t) for t in texts[:50]]  # limit to 50
    return jsonify({'results': results})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
