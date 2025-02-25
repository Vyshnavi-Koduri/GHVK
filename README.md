## **Combinational Logic Depth Prediction using Machine Learning**  

## **Overview**  
This project applies **Machine Learning (ML)** to predict the **combinational logic depth** of a circuit. Instead of running a time-consuming **RTL synthesis**, this approach estimates **timing violations** early in the VLSI design process.  

We use a **Random Forest Regressor** trained on circuit features like **fan-in, fan-out, gate count, and critical path length** to predict the combinational depth efficiently.  

## **How It Works**  
1. **Dataset Creation**: A dataset is manually created with key circuit parameters.  
2. **Feature Engineering**: Extracts important circuit features (fan-in, fan-out, etc.).  
3. **Model Training**: A **Random Forest Regressor** is trained to learn the relationship between circuit features and combinational depth.  
4. **Prediction**: The trained model predicts the **combinational depth** for new circuit signals.  
5. **Evaluation**: The model is tested and evaluated using **Mean Absolute Error (MAE)** and **R² Score** to measure accuracy.  

---

## **Code Explanation**  

### **1. Import Required Libraries**  
``` python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
```
- **pandas**: For handling tabular data.  
- **numpy**: For numerical operations.  
- **sklearn.model_selection.train_test_split**: Splits data into training & test sets.  
- **sklearn.ensemble.RandomForestRegressor**: Machine learning model used for prediction.  
- **sklearn.metrics**: Evaluates model performance.  

---

### **2. Create a Sample Dataset**  
```python
data = {
    'fan_in': [2, 1, 1, 3, 2, 4, 3],
    'fan_out': [1, 1, 1, 2, 1, 3, 2],
    'gate_count': [3, 2, 3, 4, 3, 5, 4],
    'critical_path_length': [3, 2, 3, 4, 3, 5, 4],
    'high_delay_gates': [0, 0, 0, 1, 0, 1, 1],
    'combinational_depth': [3, 2, 3, 4, 3, 5, 4]  # Target variable
}
df = pd.DataFrame(data)
```
- **Each row represents a circuit path**, and each column represents a feature of the circuit.  
- The **target variable** is `combinational_depth`, which the model will predict.  

---

### **3. Split Data into Training & Testing Sets**  
```python
X = df.drop(columns=['combinational_depth'])  # Features
y = df['combinational_depth']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- **`X`**: Circuit features (input data).  
- **`y`**: Combinational depth (output to predict).  
- **`train_test_split`**: Splits data into **70% training** and **30% testing** for evaluation.  

---

### **4. Train a Machine Learning Model**  
```python
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- Uses **Random Forest Regressor** with **100 decision trees**.  
- Trains the model on the **training dataset**.  

---

### **5. Predict on Test Data & Evaluate Model**  
```python
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")
```
- **`y_pred`**: Model predictions.  
- **`mean_absolute_error`**: Measures average error in predictions.  
- **`r2_score`**: Measures how well the model fits the data (closer to 1 is better).  

---

### **6. Predict Combinational Depth for a New Circuit**  
```python
new_signal = np.array([2, 1, 3, 3, 0]).reshape(1, -1)  # Ensuring correct shape
predicted_depth = model.predict(new_signal)
print(f"Predicted Combinational Depth: {predicted_depth[0]}")
```
- Predicts the **combinational depth** for a **new circuit signal**.  
- Uses `.reshape(1, -1)` to match the input format expected by the model.  

---

## **Installation**  
Make sure **Python 3.x** is installed. Install required dependencies using:  
```bash
pip install pandas numpy scikit-learn
```  

---

## **How to Run the Code**  
1. Clone the repository:  
   ```
   git clone https://github.com/Vyshnavi-Koduri/GHVK
   cd GHVK
   ```
2. Run the script:  
   ```
   python code.py
   ```
     

---

## **Expected Output**  
After running the script, you will see something like:  
```
Mean Absolute Error: 0.2  
R² Score: 0.95  
Predicted Combinational Depth: 3  
```
- **MAE = 0.2**: The model's predictions are very close to actual values.  
- **R² Score = 0.95**: The model explains **95% of the variance** in the data.  
- **Predicted Combinational Depth = 3**: This is the model's output for a new signal.  

---

## **Future Improvements**  
🔹 Use **real RTL-based datasets** instead of synthetic data.  
🔹 Experiment with **Neural Networks** or **Gradient Boosting** for better accuracy.  
🔹 Automate **feature extraction from Verilog code** for real-world application.  

---
