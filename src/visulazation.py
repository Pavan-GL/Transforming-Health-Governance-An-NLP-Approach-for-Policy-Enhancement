import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'visualization.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def plot_sentiment_distribution(predictions):
    try:
        logging.info("Starting sentiment distribution plot.")
        plt.figure(figsize=(8, 5))
        sns.countplot(x=predictions)
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(ticks=[0, 1], labels=['Negative', 'Positive'])
        
        output_path = 'results/sentiment_distribution.png'
        plt.savefig(output_path)
        logging.info(f"Sentiment distribution plot saved at {output_path}.")
        plt.show()
    except Exception as e:
        logging.error(f"Error while plotting sentiment distribution: {e}")

if __name__ == "__main__":
    # Example usage with dummy predictions
    predictions = [0, 1, 1, 0, 1, 0]  # Replace with actual predictions
    plot_sentiment_distribution(predictions)
