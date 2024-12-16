import pandas as pd
from data_processor import GermanHousingDataProcessor

def main():
    # Load dataset
    data = pd.read_csv("data/sample_data.csv")
    processor = GermanHousingDataProcessor(data)

    # Preprocess data
    X, y = processor.comprehensive_preprocessing()

    # Train model
    model = processor.advanced_model_training()

    # Visualize feature importance
    processor.visualize_feature_importance(model)

if __name__ == "__main__":
    main()
