import os
import logging
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import nltk

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'topic_modeling.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)

class TopicModel:
    def __init__(self, num_topics=5, passes=15):
        self.num_topics = num_topics
        self.passes = passes
        self.model = None

    def preprocess_for_topic_modeling(self, text):
        """Tokenize and preprocess the text for topic modeling."""
        return [word for word in nltk.word_tokenize(text.lower()) if word.isalpha()]

    def train(self, df):
        """Train the LDA model on the provided DataFrame."""
        try:
            logging.info("Starting preprocessing of text data.")
            df['tokens'] = df['cleaned_text'].apply(self.preprocess_for_topic_modeling)

            logging.info("Creating dictionary and corpus for LDA.")
            dictionary = corpora.Dictionary(df['tokens'])
            corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]
            
            logging.info("Training LDA model.")
            self.model = LdaModel(corpus, num_topics=self.num_topics, id2word=dictionary, passes=self.passes)
            logging.info("LDA model trained successfully.")

            return self.model, dictionary

        except Exception as e:
            logging.error(f"Error during training: {e}")
            raise

    def save_model(self, model_path, dictionary_path):
        """Save the trained model and its dictionary."""
        try:
            self.model.save(model_path)
            self.model.save(dictionary_path)
            logging.info("Model and dictionary saved successfully.")
        except Exception as e:
            logging.error(f"Error while saving model: {e}")
            raise

if __name__ == "__main__":
    try:
        logging.info("Loading training data.")
        df = pd.read_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/train_data.csv')
        
        topic_model = TopicModel(num_topics=5, passes=15)
        lda_model, dictionary = topic_model.train(df)

        # Save the model and dictionary
        model_path = 'D:/Transforming Health Governance An NLP Approach for Policy Enhancement/results/topic_model'
        dictionary_path = 'D:/Transforming Health Governance An NLP Approach for Policy Enhancement/results/dictionary.dict'
        topic_model.save_model(model_path, dictionary_path)

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
