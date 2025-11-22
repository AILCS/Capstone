# Capstone Project: Credit Risk Assessment

## Credit Risk Assessment Final Report

### **1) <ins> Define the Problem Statement**</ins>  
This project develops a machine learning model to predict a loan applicant’s Credit Risk Score using diverse demographic, financial, and loan-related features. The objective is to improve the accuracy, efficiency, and fairness of Credit Risk Assessment compared to conventional evaluation methods. The model also addresses key challenges such as feature complexity, interdependence, and the need for interpretability to ensure transparent and data-driven lending decisions.

      
### **2) <ins> Model Outcomes or Predictions**</ins>  
The proposed solution employs a supervised machine learning approach to predict the Credit Risk Score[^1] of loan applicants — a continuous measure of credit quality — making regression the appropriate modelling technique. The model incorporates regularisation methods (Ridge and Lasso regression) to enhance stability and mitigate multicollinearity. Polynomial feature transformations are applied to numerical features to capture higher-order relationships, while One-Hot Encoding is used to convert categorical attributes into a suitable numerical format. To ensure robust and reliable performance, GridSearchCV with cross-validation is utilised for systematic hyperparameter tuning and optimal predictive accuracy.

[^1]: A lower score implies a higher credit quality.


### **3) <ins> Data Acquisition**</ins>  
Data[^2] for this project is sourced from Kaggle, a reputable platform providing publicly available and well-structured datasets. Given logistical and privacy constraints, independent collection of actual credit applicant data is not feasible. The chosen dataset includes a comprehensive set of variables — such as demographic information, employment status, income, debt ratios, credit history, and loan characteristics — that reflect real-world credit assessments. **Exploratory Data Analysis (EDA)** and visualisations, including count plots for categorical features, and histograms and correlation heatmaps for numerical features, reveal meaningful relationships with the target Credit Risk Score, confirming the dataset’s suitability for developing an effective predictive model.

[^2]: https://www.kaggle.com/datasets/lorenzozoppelletto/financial-risk-for-loan-approval?select=Loan.csv

The dataset includes the following:

•	**Categorical Variables:**
1.	**ApplicationDate:** Loan application date
2.	**EmploymentStatus:** Job situation
3.	**EducationLevel:** Highest education attained
4.	**MaritalStatus:** Applicant's marital state
5.	**HomeOwnershipStatus:** Homeownership type
6.	**LoanPurpose:** Reason for loan

•	**Numerical Variables:**
1.	**Age:** Applicant's age
2.	**AnnualIncome:** Yearly income
3.	**CreditScore:** Creditworthiness score
4.	**Experience:** Work experience
5.	**LoanAmount:** Requested loan size
6.	**LoanDuration:** Loan repayment period
7.	**NumberOfDependents:** Number of dependents
8.	**MonthlyDebtPayments:** Monthly debt obligations
9.	**CreditCardUtilizationRate:** Credit card usage percentage
10.	**NumberOfOpenCreditLines:** Active credit lines
11.	**NumberOfCreditInquiries:** Credit checks count
12.	**DebtToIncomeRatio:** Debt to income proportion
13.	**BankruptcyHistory:** Bankruptcy records
14.	**PreviousLoanDefaults:** Prior loan defaults
15.	**PaymentHistory:** Past payment behaviour
16.	**LengthOfCreditHistory:** Credit history duration
17.	**SavingsAccountBalance:** Savings account amount
18.	**CheckingAccountBalance:** Checking account funds
19.	**TotalAssets:** Total owned assets
20.	**TotalLiabilities:** Total owed debts
21.	**MonthlyIncome:** Income per month
22.	**UtilityBillsPaymentHistory:** Utility payment record
23.	**JobTenure:** Job duration
24.	**NetWorth:** Total financial worth
25.	**BaseInterestRate:** Starting interest rate
26.	**InterestRate:** Applied interest rate
27.	**MonthlyLoanPayment:** Monthly loan payment
28.	**TotalDebtToIncomeRatio:** Total debt against income
29.	**LoanApproved:** Loan approval status

•	**Target Variable (Numerical):**

  -	**RiskScore: Credit Risk Assessment score**

<br>
<p align="center"><strong>Figure 1:</strong> Histogram for 30 Numerical Variables (including target)</p>
<br>
<p align="center"><img width="654" height="1454" alt="image" src="https://github.com/user-attachments/assets/1d1cb3db-4bb2-472c-b73d-120b015da0c6" /></p>

<br>
<p align="center"><strong>Figure 2:</strong> Correlation Heatmap for 30 Numerical Variables (including target)</p>
<br>
<p align="center"><img width="940" height="728" alt="image" src="https://github.com/user-attachments/assets/fbfca38e-e204-4276-aa1c-70edb6ccbf37" /></p>

<br>
<p align="center"><strong>Figure 3:</strong> Count plot for 5 Categorical Variables (excluding **ApplicationDate**)</p>  
<br>
<p align="center"><img width="940" height="415" alt="image" src="https://github.com/user-attachments/assets/29fe6ab1-0519-4e38-923f-fb7dad319824" /></p>


### **4) <ins> Data Preprocessing / Preparation**</ins>  
<ins>Data Cleaning and Feature Selection</ins>  
The dataset used in this study was complete and free of missing values, requiring no imputation. The **ApplicationDate** feature was removed, as temporal information was not considered predictive of the target variable, **RiskScore**. Categorical variables were examined for sparsely populated categories, and none were identified, eliminating the need for category consolidation prior to One-Hot Encoding. Numerical features were reviewed for outliers or invalid entries, and none were observed.
  
Correlation analysis among numerical variables revealed redundancy, leading to the removal of the following features to reduce multicollinearity: **Experience, MonthlyIncome, MonthlyLoanPayment, NetWorth, BaseInterestRate**. 
  
Additionally, features exhibiting negligible correlation with all other variables —including **NumberOfDependents, NumberOfOpenCreditLines, NumberOfCreditInquiries, PaymentHistory, SavingsAccountBalance, CheckingAccountBalance, UtilityBillsPaymentHistory, JobTenure** — were excluded.

The resulting dataset comprised:

•	**Target Variable:** 
  - **RiskScore**

•	**Numerical Features:** 
1.	**Age**
2.	**AnnualIncome**
3.	**CreditScore**
4.	**LoanAmount**
5.	**LoanDuration**
6.	**MonthlyDebtPayments**
7.	**CreditCardUtilizationRate**
8.	**DebtToIncomeRatio**
9.	**BankruptcyHistory**
10.	**PreviousLoanDefaults**
11.	**LengthOfCreditHistory**
12.	**TotalAssets**
13.	**TotalLiabilities**
14.	**InterestRate**
15.	**TotalDebtToIncomeRatio**
16.	**LoanApproved**

•	**Categorical Features:** 
1.	**EmploymentStatus**
2.	**EducationLevel**
3.	**MaritalStatus**
4.	**HomeOwnershipStatus**
5.	**LoanPurpose**
<br>

**Principal Component Analysis (PCA)** was applied to assess the potential for dimensionality reduction among the numerical features. The scree plot indicated that 12 components were necessary to capture at least 90% of the variance, suggesting minimal reduction in dimensionality. Considering the limited reduction and the loss of feature interpretability, PCA was not employed in subsequent modelling.
  
<br>  
<p align="center"><strong>Figure 4:</strong> Scree plot for Numerical Features</p>
<br>  
<p align="center"><img width="752" height="546" alt="image" src="https://github.com/user-attachments/assets/08c1c34f-ee91-4c53-9c1f-f885a7a9e5d9" /></p>

<ins>Data Splitting</ins>  
The preprocessed dataset was partitioned into training and testing sets using an 80/20 split, ensuring that sufficient data remained for model development while reserving a representative subset for evaluating model performance.

<ins>Feature Engineering and Encoding</ins>  
To enhance the predictive capability of the model, **Polynomial Feature Expansion** was applied to the numerical variables to capture potential non-linear relationships. **One-Hot Encoding** was used for categorical variables to transform them into a numerical format suitable for regression modeling. These preprocessing steps produced a clean, consistent, and analytically robust dataset, providing a strong foundation for developing an accurate and interpretable predictive model for the Credit Risk Score. 



### **5) <ins> Modelling**</ins>  



### **6) <ins> Model Evaluation**</ins>  



### **7) <ins> Deployment**</ins>  
