import csv
import os
from collections import Counter, defaultdict

class DatasetLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []
        self.entities = set()
        self.sentiments = set()
        self.loaded = False

    def load(self):
        """Parse dataset into structured records."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        self.data = []
        self.entities = set()
        self.sentiments = set()

        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 4:
                    item_id, entity, sentiment, text = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                    # Standardize sentiment label
                    sentiment = sentiment.capitalize()
                    self.data.append({
                        'id': item_id,
                        'entity': entity,
                        'sentiment': sentiment,
                        'text': text
                    })
                    self.entities.add(entity)
                    self.sentiments.add(sentiment)

        self.loaded = True
        return len(self.data)

    def get_summary(self):
        """Get dataset metrics overview."""
        if not self.loaded:
            self.load()

        sentiment_counts = Counter(item['sentiment'] for item in self.data)
        entity_counts = Counter(item['entity'] for item in self.data)

        return {
            'total_records': len(self.data),
            'unique_entities': len(self.entities),
            'sentiment_distribution': dict(sentiment_counts),
            'top_entities': entity_counts.most_common(10)
        }

    def get_entity_breakdown(self, entity_name):
        """Get sentiment distribution for a specific entity."""
        if not self.loaded:
            self.load()

        entity_items = [item for item in self.data if item['entity'].lower() == entity_name.lower()]
        counts = Counter(item['sentiment'] for item in entity_items)

        sample_texts = [
            {'text': item['text'], 'sentiment': item['sentiment']}
            for item in entity_items[:5]
        ]

        return {
            'entity': entity_name,
            'total_tweets': len(entity_items),
            'distribution': dict(counts),
            'samples': sample_texts
        }

    def get_training_samples(self, sample_size=4000):
        """Get balanced sample subset for ML training."""
        if not self.loaded:
            self.load()

        # Filter out 'Irrelevant' if desired or keep standard classes
        valid_items = [item for item in self.data if item['sentiment'] in ['Positive', 'Negative', 'Neutral'] and len(item['text']) > 10]
        
        # Sample proportionally
        import random
        random.seed(42)
        sampled = random.sample(valid_items, min(sample_size, len(valid_items)))
        return [(item['text'], item['sentiment']) for item in sampled]
