import os
import logging
import pandas as pd
from data_preprocessing import preprocess_data
from sentiment_analysis import train_model, evaluate_model
from train import TopicModel
from visulazation import plot_sentiment_distribution

# Set up logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'main_results.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    try:
        logging.info("Starting the data processing and analysis pipeline.")
        
        # # Step 1: Preprocess data
        # logging.info("Preprocessing data.")
        # train_df, test_df = preprocess_data('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/health_policies.csv')
        # logging.info("Data preprocessing completed.")

        # Step 2: Train sentiment analysis model
        train_df = pd.read_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/train_data.csv')
        test_df = pd.read_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/test_data.csv')

        logging.info("Training sentiment analysis model.")
        model = train_model(train_df)
        logging.info("Sentiment analysis model trained successfully.")

        # Step 3: Evaluate model
        logging.info("Evaluating sentiment analysis model.")
        predictions = evaluate_model(model, test_df)
        logging.info("Model evaluation completed.")

        # Step 4: Visualize results
        logging.info("Visualizing sentiment distribution.")
        plot_sentiment_distribution(predictions)
        logging.info("Sentiment distribution visualization completed.")

        # Step 5: Train topic model
        logging.info("Training topic model.")
        lda_model = TopicModel.train(train_df)
        logging.info("Topic model trained successfully.")

        # Returning or saving outputs if needed
       
        output =  {
            "train_df": train_df,
            "test_df": test_df,
            "sentiment_model": model,
            "predictions": predictions,
            "topic_model": lda_model
        }
        print(output)
        return output

    except Exception as e:
        logging.error(f"Error during execution: {e}")

if __name__ == "__main__":
    outputs = main()
    if outputs:
        logging.info("Execution completed successfully.")
        # Optionally, save outputs or perform further actions
