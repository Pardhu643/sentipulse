import csv
import os
import random
from collections import Counter

class DatasetLoader:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = []
        self.entities = set()
        self.sentiments = set()
        self.loaded = False

    def load(self, max_records=15000):
        """Memory-efficient parser loading up to max_records."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Dataset not found at {self.csv_path}")

        self.data = []
        self.entities = set()
        self.sentiments = set()

        with open(self.csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            count = 0
            for row in reader:
                if len(row) >= 4:
                    item_id, entity, sentiment, text = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                    sentiment = sentiment.capitalize()
                    
                    # Store compact tuples (id, entity, sentiment, text) to save RAM
                    self.data.append((item_id, entity, sentiment, text))
                    self.entities.add(entity)
                    self.sentiments.add(sentiment)
                    
                    count += 1
                    if max_records and count >= max_records:
                        break

        self.loaded = True
        return len(self.data)

    def get_summary(self):
        """Get dataset metrics overview."""
        if not self.loaded:
            self.load()

        sentiment_counts = Counter(item[2] for item in self.data)
        entity_counts = Counter(item[1] for item in self.data)

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

        entity_items = [item for item in self.data if item[1].lower() == entity_name.lower()]
        counts = Counter(item[2] for item in entity_items)

        sample_texts = [
            {'text': item[3], 'sentiment': item[2]}
            for item in entity_items[:5]
        ]

        return {
            'entity': entity_name,
            'total_tweets': len(entity_items),
            'distribution': dict(counts),
            'samples': sample_texts
        }

    def get_training_samples(self, sample_size=1500):
        """Get balanced sample subset for ML training."""
        if not self.loaded:
            self.load()

        valid_items = [item for item in self.data if item[2] in ['Positive', 'Negative', 'Neutral'] and len(item[3]) > 10]
        
        random.seed(42)
        sampled = random.sample(valid_items, min(sample_size, len(valid_items)))
        return [(item[3], item[2]) for item in sampled]
