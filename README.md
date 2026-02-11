# BMW Price Prediction Project

This project implements a Machine Learning model to predict the resale price of BMW cars based on various features such as model, year, mileage, and engine size. It uses a Linear Regression model trained on a provided dataset.

## Project Structure

```
Project-BMW Price Prediction/
├── core/                   # Jupyter notebooks for data analysis & model experimentation
│   ├── model_selction.ipynb
│   └── visilizing_the_data.ipynb
├── data/                   # Data directory
│   ├── bmw.csv             # Main dataset for training
│   ├── input.csv           # Input data for making predictions
│   └── output.csv          # Generated predictions
├── main.py                 # Main orchestration script (Train & Inference)
├── model.pkl               # Serialized trained model
├── pipeline.pkl            # Serialized data preprocessing pipeline
└── README.md               # Project documentation
```

## Dataset

The model is trained on `data/bmw.csv`, which contains the following columns:
- **Numerical Features:** Year, Engine_Size_L, Mileage_KM, Sales_Volume
- **Categorical Features:** Model, Region, Color, Fuel_Type, Transmission, Sales_Classification
- **Target Variable:** Price_USD

## Dependencies

Ensure you have Python installed. You can install the required libraries using pip:

```bash
pip install pandas scikit-learn joblib
```

## Usage

The `main.py` script handles both training and inference modes automatically.

### 1. Training the Model
To train the model, ensure that `model.pkl` does not exist in the root directory. The script will:
1. Load `data/bmw.csv`.
2. Preprocess the data (impute missing values, scale numerical features, and one-hot encode categorical features).
3. Train a `LinearRegression` model.
4. Save the trained model to `model.pkl` and the preprocessing pipeline to `pipeline.pkl`.

```bash
# Delete existing model if you want to retrain
rm model.pkl pipeline.pkl  # On Windows: del model.pkl pipeline.pkl

python main.py
```

### 2. Making Predictions (Inference)
If `model.pkl` exists, the script runs in inference mode:
1. Loads the saved model and pipeline.
2. Reads input data from `data/input.csv`.
3. Predicts the prices.
4. Saves the results (original input + predicted prices) to `data/output.csv`.

```bash
python main.py
```

## Methodology

### Data Preprocessing
The project uses a `scikit-learn` Pipeline with a `ColumnTransformer` to handle different data types:
- **Numerical Attributes:** Imputed with the median and scaled using `StandardScaler`.
- **Categorical Attributes:** Encoded using `OneHotEncoder`.

### Model
A simple **Linear Regression** model is used for price prediction.

## Analysis & Notebooks
The `core/` directory contains Jupyter notebooks used during the development phase:
- `visilizing_the_data.ipynb`: Exploratory Data Analysis (EDA) and visualization.
- `model_selction.ipynb`: Experiments with different models and hyperparameter tuning.
