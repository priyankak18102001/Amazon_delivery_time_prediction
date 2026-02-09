# Amazon Delivery Time Prediction

## Project Overview

This project predicts the estimated delivery time for e-commerce orders using Machine Learning.
The model analyzes factors such as distance, traffic, weather, vehicle type, delivery area, and agent details to estimate delivery time.

The final solution includes:

* Data preprocessing and feature engineering
* Exploratory Data Analysis (EDA)
* Multiple regression models
* Model tracking with MLflow
* Interactive prediction app built with Streamlit

---

## Features

* Predict delivery time based on real-world delivery conditions
* Interactive user interface using Streamlit
* Comparison of multiple regression models
* Feature importance visualization
* Model performance tracking with MLflow

---

## Technologies Used

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* MLflow
* Streamlit

---

## Dataset Description

The dataset contains delivery-related information such as:

* Agent Age and Rating
* Store and Drop Location Coordinates
* Traffic Conditions
* Weather Conditions
* Vehicle Type
* Delivery Area
* Product Category
* Delivery Time (Target Variable)

---

## Machine Learning Models Used

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Evaluation Metrics:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* R² Score

---

## Project Structure

```
Amazon_delivery_prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── delivery_model.pkl
│   ├── feature_columns.pkl
│   ├── feature_importance.csv
│
├── notebooks/
└── data/
```

---

## How to Run the Project

1. Clone the repository:

```
git clone https://github.com/priyankak18102001/Amazon_delivery_time_prediction.git
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Run the Streamlit app:

```
streamlit run app.py
```

---

## Results and Insights

Key observations from the analysis:

* Delivery time increases with distance
* Traffic conditions significantly impact delivery time
* Weather conditions affect delivery performance
* Distance is one of the most important predictive features

---

## Future Improvements

* Deploy the application online
* Add real-time traffic or weather integration
* Improve model accuracy with advanced tuning

---

## Author

Priyanka Kumawat
LinkedIn: [www.linkedin.com/in/priyanka-kumawat-7177092a3](http://www.linkedin.com/in/priyanka-kumawat-7177092a3)
