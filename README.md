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
  - I also used **Decision Trees** to gain insights into feature importance and model structure.  

- **Advanced Models:**  
  - **Random Forest** helped improve predictive performance through ensemble learning.  
  - **XGBoost** was implemented to leverage boosting for higher accuracy and recall.  
  - I experimented with **Support Vector Machines (SVM)** to evaluate non-linear decision boundaries.  

- **Hyperparameter Tuning:**  
  - I used **RandomizedSearchCV** and **GridSearchCV** to fine-tune key hyperparameters for each model.  
  - For XGBoost, I enabled **early stopping** to prevent overfitting and improve generalization.  

- **Model Evaluation:**  
  - I compared models based on **accuracy, precision, recall, F1-score, and AUC-ROC**.  
  - Since class imbalance was a concern, I paid close attention to **recall and F1-score** to ensure the minority class was well predicted.
 
### Model Evaluation Summary

Below is a summary of various models for classifying defaults, with a focus on recall and F1 score for the default (minority) class:

| Model              | Accuracy | Default Precision | Default Recall | Default F1-score |
|--------------------|----------|-------------------|----------------|------------------|
| Logistic Regression| 81.7%    | 0.28              | 0.35           | 0.31             |
| Random Forest      | 86.9%    | 0.37              | 0.18           | 0.25             |
| XGBoost            | 84.6%    | 0.32              | 0.28           | 0.30             |
| Tuned XGBoost      | 76.0%    | 0.26              | 0.58           | 0.36             |


### Summary

- **Logistic Regression:**  
  - **Accuracy:** 81.7%  
  - **Default Recall:** 35%  
  - **Default F1 Score:** 0.31  
  Achieves moderate overall accuracy with a modest recall and F1 score.

- **Random Forest:**  
  - **Accuracy:** 86.9%  
  - **Default Recall:** 18%  
  - **Default F1 Score:** 0.25  
  Shows high overall accuracy but very low default recall and F1 score.

- **Untuned XGBoost:**  
  - **Accuracy:** 84.6%  
  - **Default Recall:** 28%  
  - **Default F1 Score:** 0.30  
  Performs similarly to Logistic Regression in overall accuracy with slightly lower default recall and F1 score.

- **Tuned XGBoost:**  
  - **Accuracy:** 76.0%  
  - **Default Recall:** 58%  
  - **Default F1 Score:** 0.36  
  Despite lower overall accuracy, it delivers the highest default recall and best F1 score, making it the best model when the primary goal is to detect as many defaults as possible.

**Conclusion:**  
The **Tuned XGBoost model** is the best choice for classifying defaults because it significantly improves the detection of default cases (58% recall) while maintaining a better balance between precision and recall (F1 score of 36%), which is critical for imbalanced classification tasks.


## 7. Future Improvements

- Implementing deep learning models for enhanced prediction accuracy.
- Exploring additional data sources for better risk assessment.
- Deploying the model as a web-based or API service for real-time credit scoring.

## 8. Contributing

Contributions are welcome! Fork the repository, create a new branch, make your changes, and submit a pull request.

## 9. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
