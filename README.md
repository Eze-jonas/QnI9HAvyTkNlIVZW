#### Intelligent Product Marketing Strategy Using Machine Learning

**Context:**
* Project completed while at Apziva. This repository contains exploration, preprocessing, model development and Shap computation and analysis.

This project aims to:
* predict which customers are likely to subscribe to a term deposit
* improve the efficiency of a previously random search for product subscription
* identify customers most likely to subscribe, preventing calling customers who never going to subscribe 
* segmenting customer to identify subscribers attributes and optimize marketing of additional investment products

 The analysis was performed using a European bank’s marketing dataset, generated from its telephone call campaigns.

This work highlights how machine learning can:

* Improve the efficiency of a random product marketing strategy, enabling companies to acquire subscribers with less effort and cost.

* identify customers most likely to subscribe, supporting a more targeted and data-driven marketing approach

* segmenting customer to identify subscribers attributes and optimize marketing of additional investment products

 All notebooks are reproducible, auditable, and designed to generate artifacts that can be integrated into a production pipeline.  
## Table of contents
1. Project summary
2. project Achievement
3. Highlights & dataset
6. Step-by-step project flow
7. Technical details & explanations
8. How to reproduce locally
9. Repository structure
10. What I contributed / hiring-manager summary
11. Next steps & suggestions
---

#### Project Summary

This project used a European bank’s marketing dataset, generated from its telephone call campaigns, to build classification models that predict whether a customer will subscribe to a term deposit product.

* Improved the efficiency of the previously random campaign through a pre-campaign model

* Identified customers most likely to subscribe via a post-campaign model

* segmented customers to identify subscribers attributes and optimize marketing of additional investment products through k-means clustering

The work included exploratory data analysis (EDA), robust preprocessing using a reusable pipeline, model selection and hyperparameter tuning, and the creation of saved preprocessing artifacts for reproducibility.

#### Project Achievements

Developed a two-stage machine learning model and Customer segmentation 

* Pre-campaign model

* Post-campaign model

* Customer Segmentation

#### Pre-Campaign Model
This was developed by dropping the campaign(call) related features

Three different ensemble models — XGBoost, LightGBM (LGBM), and Random Forest — were experimented with and evaluated.

Model Performance on the Test Set and minority class:

**XGBoost:** Achieved an efficiency of 8.48%, captured 84.1% of potential subscribers, and reduced unnecessary calls by 11,437 (28.59%), saving approximately 809.6 hours of call time.

**LightGBM:** Achieved an efficiency of 9.23%, captured 70.34% of potential subscribers, and reduced unnecessary calls by 18,071 (45.18%), saving approximately 1,279.1 hours of call time.

**Random Forest:** achieved efficiency of 8.5%, captured 79.31% potential subscribers and reduced unnecessary calls by 12,746(31.86%), saving approximately 902.2 hours of call time.

#### Post-campaign model
This was developed with all the features including the campaign(call) related features using LightGBM model that achieved:
* efficiency of 48%
* identified 72% of customers most likely to subscribe.
* area under the curve(AUC) value of 94%
  
#### Customer Segmentation
This was done using K-means clustering that segmented the customers into two clusters; 0 and 1

**Cluster 0** consists of older customers aged between 41 and 70+. About 62% of them have a high account balance (greater than €407), while 37% have a low account balance (less than or equal to €407). 67% are married, 11% are single, and 21% are divorced. Around 12% are retired. Only 41% of this group showed interest in a housing loan.

**Cluster 1** consists of younger customers aged approximately 19 to 41 years (with ages below 22 compressed to 22 due to winsorization). About 55% of them have a high account balance (greater than €407), while 45% have a low account balance (less than or equal to €407). 40% are married, 51% are single, and 0.8% are divorced. Around 0.1% are retired and 57% of this group showed interest in a housing loan.

## Highlights & dataset
* The project used term-deposit-marketing-2020.cv(not included in this repo for privacy) from a European bank’s marketing dataset, generated from its telephone call campaigns. 
* The dataset has 40,000 rows and 14 columns:
* y – binary target (customer subscribed = yes, not subscribed = 0)
* Notebooks are written to work with this structure, so anyone supplying a similarly formatted CSV can reproduce the analysis.

### Step-by-step project flow

Below is a concise, numbered flow showing how the work progresses from raw data to a selected model.

1. **Exploratory Data Analysis (EDA)**
During EDA,the data types of numerical variables were converted from integers to floats. New features were created, and the linear relationships and distributions of the variables were examined. Skewness was detected and handled using log and Yeo-Johnson transformation methods. Outliers were identified and treated using the Winsorization method. Numerical features were normalized using the Z-score standardization method, and categorical variables were encoded.
2. **Preprocessing & Pipeline development**  
* Built a sklearn.Pipeline with small, composable custom transformers to:  
* convert numerical variables data types to float  
* creat new features; age_group and balance_group from age and balance column respectively
* drop age column
* transfomed using Yeo-Johnson
* apply winsorization (to cap extreme outliers)  
* standardize features using z-score (StandardScaler)
* One hot encoded categorical veriables and drop their first dummy.  
* The pipeline is saved as dynamic_customer_pipeline.pkl and the transformed DataFrame is saved as df_ML1.csv for reproducibility.

3. **Modeling & model selection**
* The dataset was split into 70% for training, 15% for validation, and 15% for testing.
* Manual class weights were applied to address the class imbalance issue. The baseline class weight (12.8) was computed from the training set, and additional values were tweaked around this baseline for experimentation and final selection.
* Three random state seeds were auto-generated using random.randint(1000, 9999) and tested for stability.
* 5-fold cross-validation within Hyperopt CV was employed for hyperparameter tuning, using a custom scoring metric that weighted precision (40%) and recall (60%). A total of 30 hyperparameter configurations were tested, and the best-performing candidate model was selected for final training.
  
 **XGBoost, LightGBM and Random Forest** were trained with the best parameter combination and evaluated using classification report and AUC.
 LGBM performed better and was selected for further training and analysis.

4. **Trained and Evaluated the selected LGBM on the full Dataset for SHAP analysis**
This helped to understand the features that pushed the baseline probabilty up and once that pushed it down.

## Technical details & explanations

**why custom class weight?**

Custom class weight handls class imbalance issue

**why Yeo Johnson Transformation?**

Yeo Johnson transformation helps to reduce skeweness of a data with 0 or negative data points. 

**Why winsorize?** 

Winsorization reduces the influence of extreme outliers by capping values at a defined percentile. This helps models avoid being skewed by a few extreme responses.

**Why z-score standardization?** 

Standardization (z-score) centers features to mean 0 and unit variance which helps distance-based and regularized models converge and perform consistently across scales.

**why One hot encod**

One hot encode helps convert categorical variables to numeric 

**Why 5-Fold hyperop CV?** 

5-fold divide the training set into 5 equalsub sets, then the model is trained and evaluated on the subsets for good fit.
hyperopt ensures a disciplined hyperparameter search across parameter range.

**why DuckDB?** 

This was used to extract only the customers that subscribed to the term deposit for clustering analysis.


## How to Reproduce Locally (Quick Start)

* These notebooks were developed with Python 3.10 and Jupyter. Paths inside the notebooks may need adjusting depending on your setup.

 *Clone the repository

* git clone https://github.com/Eze-jonas/QnI9HAvyTkNlIVZW.git
* cd QnI9HAvyTkNlIVZW


* (Recommended) Create a virtual environment and install dependencies

* python -m venv .venv
* source .venv/bin/activate    # macOS / Linux
* .\.venv\Scripts\activate     # Windows

* pip install --upgrade pip
* pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost lightgbm scipy


### **Dataset setup**

* The project originally used an internal company dataset (term-deposit-marketing 2020.csv).

* For confidentiality reasons, the dataset is not included in this repo.

* To run the notebooks yourself, provide a dataset with the same structure:

* age : age of customer (numeric)

* job : type of job (categorical)

* marital : marital status (categorical)

* education (categorical)

* default: has credit in default? (binary)

* balance: average yearly balance, in euros (numeric)

* housing: has a housing loan? (binary)

* loan: has personal loan? (binary)

* contact: contact communication type (categorical)

* day: last contact day of the month (numeric)

*  month: last contact month of year (categorical)

* duration: last contact duration, in seconds (numeric)

* campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)

* Output (desired target):

* y - has the client subscribed to a term deposit? (binary)

* Update the file_path variable inside the notebooks to point to your dataset.

## Repository structure:
* ivFuindSYvzqwbHB/
* Exploratory_Data_Analysis.ipynb
* pre_campaign_preprocessing_pipeline.ipynb
* Pre-campaign Model_Development.ipynb
* post_campaign_preprocessing_pipeline.ipynb
* post-campaign Model_Development.ipynb
* Customer_Segmentation.ipynb
* Project_Summary.ipynb
* README.md  (this file)

### **What I Contributed — Executive Summary**
1. End-to-end ML workflow ownership: Performed EDA, designed and implemented a reusable preprocessing pipeline, applied Shap analysis, trained models, carried out k-means clustering analysis, tuned hyperparameters, and produced reproducible artifacts.
3. Production-aware design: Built custom scikit-learn transformers and pipelines that can be serialized (dynamic_customer_pipeline.pkl) and integrated into downstream systems.
4. Rigorous evaluation: Applied 5-fold cv within hyperopt, reporting  multiple metrics classification report and AUC for defensible model comparisons.
5. Results-driven experimentation:improved efficiency of the random search campaign, Identified numbers of pontential subscribers, numbers of calls need to make, numbers and percentage of unneccessary calls saved, numbers of saved hours of calls, identified customer most likely to subscribe and segmented customers. 
6. Reproducibility & documentation: Delivered well-structured, annotated notebooks so reviewers can step through decisions, validate methodology, and reproduce results.
--- 
#### Conclusion / Recommendations.
The application of intelligent modeling techniques to the product marketing campaign successfully improved the efficiency of the previously random search campaign through a pre-campaign model, identified customers most likely to subscribe via a post-campaign model, and provided the most effective strategies for marketing additional products through k-means clustering.

Integrating the **pre-campaign LightGBM model** into the company’s production system would enable the identification of 70.34% of potential subscribers, reduce unnecessary calls by 18,071 (45.18%), and save approximately 1,279.1 hours of call time. Similarly, integrating the **post-campaign LightGBM model** would ensure that 72% of contacted customers are those most likely to subscribe.


**Cluster 0** consists of older customers aged between 41 and 70+. About 62% of them have a high account balance (greater than €407), while 37% have a low account balance (less than or equal to €407). 67% are married, 11% are single, and 21% are divorced. Around 12% are retired. Only 41% of this group showed interest in a housing loan.

Given their financial stability and life stage, this group is more inclined toward low-risk, income-generating, and capital-preserving products. Suitable offerings include term deposits, government or corporate bonds, retirement and pension plans, life or health insurance products, and personalized wealth management or estate planning services. Marketing strategies for this segment should emphasize financial security, stable income, and long-term comfort, rather than aggressive loan-based products.
Cluster 1 consists of younger customers aged approximately 19 to 41 years (with ages below 22 compressed to 22 due to winsorization). About 55% of them have a high account balance (greater than €407), while 45% have a low account balance (less than or equal to €407). 40% are married, 51% are single, and 0.8% are divorced. Around 0.1% are retired and 57% of this group showed interest in a housing loan.

**Cluster 1** consists of younger customers aged approximately 19 to 41 years (with ages below 22 compressed to 22 due to winsorization). About 55% of them have a high account balance (greater than €407), while 45% have a low account balance (less than or equal to €407). 40% are married, 51% are single, and 0.8% are divorced. Around 0.1% are retired and 57% of this group showed interest in a housing loan.

Given their younger age, career stage, and growing financial activity, this group is more inclined toward moderate-risk, growth-oriented financial products that support wealth accumulation and lifestyle advancement. Suitable offerings include flexible savings plans, entry-level mutual funds or ETFs, home and personal loans, and retirement starter plans. Insurance products such as health and income protection plans are also relevant at this life stage. Marketing strategies for this segment should emphasize financial growth, independence, and convenience—highlighting digital banking solutions, automated investing, and opportunities to build assets early in life.


```python

```
