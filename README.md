# Credit Risk Assessment: Predicting Loan Defaults

## 1. Introduction

This project aims to build a Credit Risk Assessment system to help financial institutions evaluate the risk of loan applicants. Effective credit risk assessment is crucial in minimizing loan defaults, improving lending decisions, and ensuring financial stability.

## 2. Problem Statement

Financial institutions face challenges in accurately assessing borrowers' creditworthiness. Poor risk evaluation can lead to high default rates, financial losses, and an unstable credit market. This project leverages machine learning techniques to predict credit risk, enhance decision-making, and minimize potential losses.

## 3. Objectives

This project aims to achieve the following goals:

- **Analyzing Historical Loan and Credit Data** – Identify patterns and trends in borrower behavior.
- **Implementing Machine Learning Models for Risk Prediction** – Develop and train models such as Logistic Regression, Random Forest, and XGBoost to classify loan applicants as low-risk or high-risk.
- **Evaluating Model Accuracy and Optimizing Performance** – Assess model performance using key metrics and fine-tune hyperparameters to improve prediction accuracy.

## 4. Data Collection

### Data Source

The dataset used for this project can be sourced from:

- **Public Credit Risk Datasets:**
  - Kaggle Lending Club loan data

### Features in the Dataset  

The dataset consists of **255,347** loan applications with **18 features**, categorized into different aspects of a borrower's financial profile.  

**1. Demographic Information**  
- **Age** – The borrower's age.  
- **Education** – Highest level of education attained.  
- **EmploymentType** – Type of employment (e.g., Salaried, Self-Employed).  
- **Income** – The borrower’s annual income.  
- **MaritalStatus** – Indicates if the borrower is Single, Married, etc.  
- **HasDependents** – Whether the borrower has dependents.  

**2. Credit History**  
- **CreditScore** – A numerical representation of the borrower’s creditworthiness.  
- **NumCreditLines** – Number of existing credit lines.  

**3. Loan Details**  
- **LoanAmount** – The total amount borrowed.  
- **InterestRate** – The percentage interest charged on the loan.  
- **LoanTerm** – The duration of the loan in months.  
- **LoanPurpose** – The purpose for which the loan was taken (e.g., home, education, business).  

**4. Financial Status & Risk Indicators**  
- **MonthsEmployed** – Number of months the borrower has been employed.  
- **DTIRatio** – Debt-to-Income Ratio (Total Debt / Income).  
- **HasMortgage** – Indicates if the borrower has an existing mortgage.  
- **HasCoSigner** – Whether the loan has a co-signer.  

**5. Loan Default Indicator**  
- **Default** (Target Variable) – A binary indicator (0 = No Default, 1 = Default).  


## 5. Exploratory Data Analysis

### Summary Statistics  

The table below provides summary statistics for the numerical features in the dataset:

| Feature          | Count      | Mean      | Std Dev   | Min   | 25%   | 50%   | 75%   | Max   |
|-----------------|-----------|----------|----------|-------|-------|-------|-------|-------|
| **Age**         | 255,347   | 43.50    | 14.99    | 18    | 31    | 43    | 56    | 69    |
| **Income**      | 255,347   | 82,499   | 38,963   | 15,000 | 48,825 | 82,466 | 116,219 | 149,999 |
| **LoanAmount**  | 255,347   | 127,578  | 70,841   | 5,000 | 66,156 | 127,556 | 188,985 | 249,999 |
| **CreditScore** | 255,347   | 574.26   | 158.90   | 300   | 437   | 574   | 712   | 849   |
| **MonthsEmployed** | 255,347 | 59.54  | 34.64    | 0     | 30    | 60    | 90    | 119   |
| **NumCreditLines** | 255,347 | 2.50   | 1.12     | 1     | 2     | 2     | 3     | 4     |
| **InterestRate** | 255,347  | 13.49   | 6.64     | 2.00  | 7.77  | 13.46  | 19.25  | 25.00  |
| **LoanTerm**    | 255,347   | 36.03   | 16.97    | 12    | 24    | 36    | 48    | 60    |
| **DTIRatio**    | 255,347   | 0.50    | 0.23     | 0.10  | 0.30  | 0.50  | 0.70  | 0.90  |
| **Default**     | 255,347   | 0.12    | 0.32     | 0     | 0     | 0     | 0     | 1     |

---

**Key Insights**  

- **Age & Income**: Borrowers are mostly middle-aged (**median: 43**), with income widely spread (**median: 82,499**, max: **149,999**).  
- **Loan Amount & Credit Score**: Median loan amount is **127,556**, with a moderate credit score (**median: 574**). **25% have scores below 437**, indicating higher risk.  
- **Loan Terms & Interest Rates**: Most loans are for **24-48 months**, with interest rates reaching **25%** for riskier applicants.  
- **Debt-to-Income Ratio (DTI)**: **Median DTI is 0.50**, meaning half of borrowers allocate **50% of income** to debt repayment. **DTI > 0.70** signals higher default risk.  

### Understanding Data Distribution:

**Imbalance in distribution:** 

![image](https://github.com/user-attachments/assets/6f6d1e1b-fe71-4e75-bf1d-0f10506c5136)

- Non-defaulters make up **88.4%** of the dataset, while defaulters account for only **11.6%**.
- This class imbalance may affect model performance, requiring techniques like oversampling, undersampling, or adjusting class weights.  

![image](https://github.com/user-attachments/assets/5ed9d880-2435-4f88-92c9-a8c04e9a4062)

### KDE Plots of Numerical Features by Default Status

![image](https://github.com/user-attachments/assets/ff388dd5-b09b-4dba-bf27-e215b91e931c)

 **Insights from KDE Plots**

- **Younger borrowers (20-30)** are more likely to default, while non-defaulters are evenly spread across ages.  
- **Lower income** and **higher loan amounts** increase the likelihood of default.  
- **Defaulters have lower credit scores**, while non-defaulters generally have scores above 700.  
- **Shorter employment duration** is linked to higher default rates.  
- **Higher interest rates** and **higher Debt-to-Income (DTI) ratios** correlate with more defaults.  
- **Loan terms appear fixed at certain intervals**, with non-defaulters favoring shorter terms.  

### Correlation Analysis

![image](https://github.com/user-attachments/assets/a6d39d2a-84b7-40da-968d-ffc3d003d3ee)

**Key Takeaways from Correlation Analysis**

- **Age has the strongest negative correlation** (-0.168), meaning younger borrowers are more likely to default.  
- **Interest Rate has the strongest positive correlation** (0.131), meaning higher interest rates correspond to more defaults.  
- **Higher Loan Amounts** (0.087) slightly increase default risk due to a higher repayment burden.  
- **Longer Employment Duration** (-0.097) reduces default risk, suggesting that job stability helps prevent defaults.  
- **Higher Income** (-0.099) is associated with lower default risk, indicating that financially stable borrowers are less likely to default.  
- **Credit Score** (-0.034) shows a weak negative correlation, meaning higher scores slightly reduce default probability.  
- **Most correlations are weak to moderate**, implying that **default is influenced by multiple factors rather than a single one**.

![image](https://github.com/user-attachments/assets/81ce3551-bc33-46b4-879d-b75ab026f97f)

![image](https://github.com/user-attachments/assets/8e71c06e-ed07-4066-8956-1eca35998ad3)

![image](https://github.com/user-attachments/assets/188d25e5-3f59-4cfa-b4d1-c1216b38a9da)

![image](https://github.com/user-attachments/assets/da9944ed-bc44-4101-a997-b5374b12b4b4)

![image](https://github.com/user-attachments/assets/49351ed3-f347-402b-9ce6-ce0d9ec76074)

### Data Preprocessing

- **Class Imbalance Handling:**  
  - I addressed class imbalance using **SMOTE** to oversample the minority class and experimented with **class weighting** in model training.  

- **Feature Engineering:**  
  - Categorical variables were encoded using **One-Hot Encoding (OHE)** and **Label Encoding** where appropriate.  
  - Numerical features were scaled using **StandardScaler** to ensure consistency across different models.
  - Missing values were handled by dropping the rows with missing values, since the percentage of missing data was negligible. 

## 6. Model Development  

- **Baseline Models:**  
  - I started with **Logistic Regression** to establish a simple benchmark. 

- **Advanced Models:**  
  - **Random Forest** helped improve predictive performance through ensemble learning.  
  - **XGBoost** was implemented to leverage boosting for higher accuracy and recall.  

- **Hyperparameter Tuning:**  
  - I used **RandomizedSearchCV** and **GridSearchCV** to fine-tune key hyperparameters for each model.  

- **Model Evaluation:**  
  - I compared models based on **accuracy, precision, recall, F1-score, and AUC-ROC**.  
  - Since class imbalance was a concern, I paid close attention to **recall and F1-score** to ensure the minority class was well predicted.
 
## 7. Model Evaluation Summary  

### Baseline Model: Logistic Regression 
- **Initial Performance:**  
  - Accuracy: **73%**  
  - Recall (Defaulters): **0.54** (Low)  
  - F1-score (Defaulters): **0.31**  
- **Key Issues:**  
  - Class imbalance led to poor recall for defaulters.  
  - Majority class (non-defaulters) was predicted well, but many defaulters were missed.  

- **Improvements Applied:**  
  - Applied **class weighting (`balanced`)**.  
  - Used **SMOTE** to oversample the minority class.  
  - Tuned **C** and solver (`saga`).  

- **Final Performance (After Tuning & SMOTE):**  
  - Recall (Defaulters): **0.70** ✅  
  - F1-score (Defaulters): **0.34**  
  - Accuracy: **68%** (Reduced but recall improved)  

---

### Advanced Models & Performance 

| Model        | Accuracy | Precision (Defaulters) | Recall (Defaulters) | F1-score (Defaulters) |
|-------------|----------|-----------------------|---------------------|----------------------|
| **Logistic Regression (Tuned)** | 68% | 0.22 | **0.70** | 0.34 |
| **Random Forest** | 84% | 0.31 | 0.29 | 0.30 |
| **XGBoost** | 83% | 0.30 | 0.33 | 0.32 |
| **LightGBM** | **87%** | **0.38** | 0.16 | 0.23 |

- **Observations:**  
  - **Random Forest & XGBoost:** Higher accuracy (**83-84%**), but **low recall (~0.30)**.  
  - **LightGBM:** Best accuracy (**87%**), but struggled with recall (**0.16**).  
  - **Logistic Regression (Tuned):** Lower accuracy (**68%**) but best recall (**0.70**), making it the most effective for capturing defaulters.  

---

### Hyperparameter Tuning & Techniques Applied**  
- **Logistic Regression:** `C`, `class_weight`, `solver`  
- **Random Forest & XGBoost:** `n_estimators`, `max_depth`, `min_samples_split`, `learning_rate`  
- **LightGBM:** `boosting_type`, `num_leaves`, `feature_fraction`, `learning_rate`  

---

### Final Model Selection & Next Steps 
- **Best Model for Defaulter Prediction:** ✅ **Logistic Regression (Tuned)**  
  - Highest **recall (0.70)** ensures most defaulters are captured.  
  - Performance improved with **SMOTE + class weighting**.  
  - Lower accuracy but better at detecting the minority class.  

- **Next Steps:**  
  - Explore **ensemble learning (stacking models)** for better trade-offs.  
  - Implement **cost-sensitive learning** for further improvements.  
  - Test **threshold tuning** for better precision-recall balance.  

---
## 7. Future Improvements

- Implementing deep learning models for enhanced prediction accuracy.
- Exploring additional data sources for better risk assessment.
- Deploying the model as a web-based or API service for real-time credit scoring.

## 8. Contributing

Contributions are welcome! Fork the repository, create a new branch, make your changes, and submit a pull request.

## 9. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
