# RealEstate-Loans-Stocks-Predictions
This project applies machine learning to predict house prices, loan approvals, and stock market trends. Using datasets from real estate, financial lending, and stock markets, it builds predictive models for data-driven decision-making.

## [Demo Link](https://realestate-loans-stocks-predictionsgit-pzxbuwpqm5pkxusqgrnzjq.streamlit.app/)

## Project Structure

### Directory Explanation

- `app.py` and `home.py`: Main application files.
- `requirements.txt`: Python packages required for the project.
- `.streamlit/`: Streamlit configuration files.
- `house/`: Code and data related to house price prediction.
  - `correlation.py`: Analyzes data correlation.
  - `data.py`: Data processing script.
  - `Dataset1_Cleaned.csv`, `Dataset1.csv`: Real estate datasets.
  - `multiple_linear.py`: Multiple linear regression model.
  - `outliers.py`: Outlier detection.
- `loan/`: Code and data related to loan approval prediction.
  - `data_preprocessing.py`: Data preprocessing script.
  - `Dataset2.csv`: Loan dataset.
  - `evaluation.py`: Model evaluation script.
  - `model_training.py`: Model training script.
  - `normalization.py`: Data normalization script.
- `stock/`: Code and data related to stock market trend prediction.
  - `StockData.csv`: Stock market dataset.
  - `cnn.py`: Convolutional Neural Network model.
  - `gru.py`: Gated Recurrent Unit model.
  - `lstm.py`: Long Short-Term Memory model.
  - `preprocessing.py`: Data preprocessing script.
  - `rnn.py`: Recurrent Neural Network model.
  - `transformer.py`: Transformer model.
  - `conclusion.py`: Conclusion and result analysis.

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/yourusername/RealEstate-Loans-Stocks-Predictions.git
    ```
2. Navigate to the project directory:
    ```sh
    cd RealEstate-Loans-Stocks-Predictions
    ```
3. Install the required Python packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the main application:
    ```sh
    streamlit run app.py
    ```
2. Select different predictive models and datasets for analysis as needed.
