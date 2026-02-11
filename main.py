import os
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_att, cat_att):
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_att),
        ("cat", cat_pipeline, cat_att)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE
    df = pd.read_csv("data/bmw.csv")

    # Split data
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df.Model):
        train_df = df.loc[train_index].copy()

    bmw_label = train_df["Price_USD"].copy()
    bmw_feature = train_df.drop("Price_USD", axis=1)

    num_attributes = bmw_feature.drop(
        ["Model", "Region", "Color", "Fuel_Type", "Transmission", "Sales_Classification"],
        axis=1).columns.tolist().copy()
    cat_attributes = ["Model", "Region", "Color", "Fuel_Type", "Transmission", "Sales_Classification"]

    # Time to build Pipeline
    pipeline = build_pipeline(num_att=num_attributes, cat_att=cat_attributes)

    bmw_prepared = pipeline.fit_transform(bmw_feature)

    # Train model
    model = LinearRegression()
    model.fit(bmw_prepared, bmw_label)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Congratulations!!\nModel Train. Model saved to model.pkl")

else:
    # INFERENCE PHASE
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("data/input.csv")
    transformed_input = pipeline.transform(input_data)
    prediction = model.predict(transformed_input)
    input_data["Prediction_Price_USD"] = prediction

    input_data.to_csv("data/output.csv", index=False)

    print("Congratulations!!\nInference complete. Results saved to output.csv")
