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

### **Key Insights**  

1. **Age Distribution**  
   - The borrowers’ ages range from **18 to 69 years**, with a median age of **43**.  
   - 75% of borrowers are **below 56 years**, suggesting a middle-aged borrowing population.  

2. **Income Levels**  
   - The average income is **82,499**, but the distribution is quite spread, with a standard deviation of **38,963**.  
   - The income distribution is skewed, with the lowest earners at **15,000** and the highest at **149,999**.  

3. **Loan Amount & Credit Score**  
   - The median loan amount is **127,556**, with a maximum of **249,999**.  
   - Borrowers generally have a **moderate credit score** (median: **574**), with values ranging from **300 to 849**.  
   - A quarter of borrowers have a **credit score below 437**, indicating a segment with potentially higher risk.  

4. **Loan Terms & Interest Rates**  
   - The median loan term is **36 months**, with the majority falling between **24 and 48 months**.  
   - Interest rates vary significantly, with the highest at **25%**, possibly for riskier applicants.  

5. **Debt-to-Income Ratio (DTI)**  
   - The median **DTI ratio is 0.50**, meaning that **half of the borrowers allocate 50% of their income to debt repayment**.  
   - Borrowers with **DTI > 0.70** may be at higher risk of default due to excessive debt obligations.  

6. **Default Rate**  
   - Only **11.6% (mean: 0.12)** of the borrowers have defaulted, suggesting a predominantly creditworthy population.  
   - Most borrowers **(75%) have not defaulted**, but identifying patterns in the **11.6% who did** can help refine risk assessment models.  


### Understanding Data Distribution:

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



- **Correlation Analysis:**
  - Identifying relationships between different variables.
- **Class Imbalance Handling:**
  - Addressing imbalanced datasets using techniques like SMOTE or weighted classification.
- **Feature Engineering:**
  - Creating new features and transforming existing ones for better model performance.

## 6. Model Development

- **Baseline Models:**
  - Logistic Regression
  - Decision Trees
- **Advanced Models:**
  - Random Forest
  - XGBoost
  - Neural Networks (Optional)
- **Model Evaluation Metrics:**
  - Accuracy
  - Precision, Recall, and F1-score
  - ROC-AUC Score

## 7. Future Improvements

- Implementing deep learning models for enhanced prediction accuracy.
- Exploring additional data sources for better risk assessment.
- Deploying the model as a web-based or API service for real-time credit scoring.

## 8. Contributing

Contributions are welcome! Fork the repository, create a new branch, make your changes, and submit a pull request.

## 9. License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
