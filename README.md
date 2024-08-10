# Stock Market Sentiment Analysis

This project aims to predict stock market trends by analyzing sentiment from financial news and Reddit data, integrating various machine learning models, including GRU, LSTM, SVM, and linear regression.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Downloading FinBERT Model](#downloading-finbert-model)
- [Usage](#usage)
- [Models and Evaluation](#models-and-evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Project Overview

This project is designed to predict stock market trends by analyzing sentiment in financial news and Reddit posts. It uses FinBERT, a specialized BERT model for financial sentiment analysis, and integrates this sentiment data with various machine learning models to predict stock prices.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/XIXI112233444/Stock-Prediction
   ```
   After uploading your project, replace the placeholder URL above with your actual GitHub repository URL.

2. Navigate to the project directory:
   ```bash
   cd your-repository
   ```

3. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Downloading FinBERT Model

This project relies on the [FinBERT](https://huggingface.co/ProsusAI/finbert) model for sentiment analysis. You can download the model using the following code snippet:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

Alternatively, if you have downloaded the model manually, place it in the `./finbert_model/` directory.

## Usage

### Running Sentiment Analysis

To analyze sentiment in financial news data, use the following command:

```bash
python main.py --input path_to_your_input_file.csv
```

This command will process the news data, analyze sentiment using FinBERT, and save the results to a CSV file.

### Training Models

The project includes several models for predicting stock prices based on sentiment data:

- GRU (Gated Recurrent Unit)
- 12-layer GRU
- LSTM (Long Short-Term Memory)
- SVM (Support Vector Machine)
- Linear Regression

To train these models, ensure you have the processed sentiment data, then execute:

```bash
python main.py
```

The script will train the models, save their parameters, and evaluate their performance.

### Model Evaluation

Evaluation results, including metrics like MSE, RMSE, MAE, R², MAPE, and direction accuracy, will be logged during training. Additionally, the script generates plots comparing actual versus predicted values and the error distribution across different models.

## Models and Evaluation

The following models are trained and evaluated as part of this project:

- **GRU and LSTM**: Deep learning models suited for sequential data.
- **SVM**: A machine learning model that uses hyperplane separation for regression tasks.
- **Linear Regression**: A simple linear approach for modeling the relationship between a dependent variable and one or more independent variables.

Each model's performance is evaluated using several metrics, including mean squared error (MSE), mean absolute error (MAE), and direction accuracy.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b my-feature-branch`.
3. Make your changes and commit them: `git commit -m 'Add new feature'`.
4. Push to the branch: `git push origin my-feature-branch`.
5. Submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to [Hugging Face](https://huggingface.co/) for providing the FinBERT model, and to all the open-source contributors who made this project possible.
