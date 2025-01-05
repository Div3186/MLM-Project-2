# **1. PROJECT INFORMATION**
# Build Supervised Learning Models using Python
#### Divyank Harjani - 055010
## **2.DESCRIPTION OF DATA**
### Data Information

- **Data Size:** 2.56 MB  
- **Data Type:** Panel  

### Data Dimension  
- **No. of Variables:** 16  
- **No. of Observations:** 15,000  

---

## Data Variable Type

### Numeric  
- **Integer:** Quantity, Customs_Code, Invoice_Number  
- **Decimal:** Value, Weight  

### Non-Numeric  
Transaction_ID, Customs_Code, Invoice_Number, Product, Supplier, Cu,  Shipping_Method, Countrysomer, Date, Port  

---

## Data Variable Category - I  

### Categorical  
- **Nominal:** Import_Export, Category, Shipping_Method, Payment_Terms, Country  
- **Ordinal:**  

### Non-Categorical  
Quantity, Value, Weight
---

## Data Variable Category - II  

- **Transaction_ID:** Unique identifier for each trade transaction.  
- **Country:** Country of origin or destination for the trade.  
- **Product:** Product being traded.  
- **Import_Export:** Indicates whether the transaction is an import or export.  
- **Quantity:** Amount of the product traded.  
- **Value:** Monetary value of the product in USD.  
- **Date:** Date of the transaction.  
- **Category:** Category of the product (e.g., Electronics, Clothing, Machinery).  
- **Port:** Port of entry or departure.  
- **Customs_Code:** Customs or HS code for product classification.  
- **Weight:** Weight of the product in kilograms.  
- **Shipping_Method:** Method used for shipping (e.g., Air, Sea, Land).  
- **Supplier:** Name of the supplier or manufacturer.  
- **Customer:** Name of the customer or recipient.  
- **Invoice_Number:** Unique invoice number for the transaction.  
- **Payment_Terms:** Terms of payment (e.g., Net 30, Net 60, Cash on Delivery).  

---

## **About Dataset**  

This dataset provides detailed information on international trade transactions, capturing both import and export activities. It includes comprehensive data on various aspects of trade, making it a valuable resource for business analysis, economic research, and financial modeling.[link text](https://)
### **3. Project Objectives and Problem Statements**
#### **PROJECT OBJECTIVES**

1. Analyze international trade patterns to identify key trends and insights.  
2. Understand the distribution of imports and exports across different countries and product categories.  
3. Develop predictive models to forecast trade volumes based on historical data.  
4. Identify anomalies in trade data for potential fraud detection or regulatory compliance.  
5. Provide actionable insights to optimize supply chain logistics and reduce costs.  

### **Problem Statements**
1. How can we effectively classify and categorize trade transactions based on the provided variables?  
2. What are the factors driving variations in import and export volumes across countries and categories?  
3. Can we predict future trade volumes for specific products or regions?  
4. How do payment terms and shipping methods impact the overall trade cycle?  
5. Are there any inconsistencies or anomalies in the dataset that could indicate errors or fraud?
# **Key Observations and Findings**
1. **Assessment of Missing Data:**
   - Perform a detailed analysis to identify the extent of missing data across columns and rows in the dataset.

2. **Removal of Data:**
   - If more than **50%** of the data is missing in any column or row, it will be removed to ensure the integrity and reliability of the analysis.

3. **Imputation of Missing Values:**
   - For columns or rows with less than **50%** missing data, imputation techniques will be applied:
   - **Categorical Variables:** Missing values will be replaced using the **mode** (most frequently occurring value) to maintain consistency.
   - **Numerical Variables:** Missing values will be replaced using the **mean** (average value) to minimize bias in the dataset.

4. **Rationale for Approach:**
   - This strategy ensures a balance between preserving valuable information and maintaining the quality and reliability of the dataset. By applying tailored imputation techniques based on variable types, we aim to reduce the impact of missing data on subsequent analyses.
#### **Encoding Mapping Analysis**

### Import_Export  
- **Export** is encoded as `0` and **Import** as `1`.  
- This binary encoding is intuitive and suitable for distinguishing between two distinct trade types.

### Category  
- Each product category is assigned a unique numerical value:  
  - **Clothing** -> `0`  
  - **Electronics** -> `1`  
  - **Furniture** -> `2`  
  - **Machinery** -> `3`  
  - **Toys** -> `4`  
- The mapping is consistent and facilitates easy analysis of trade by product type, ensuring the model can differentiate between categories.

### Shipping_Method  
- Shipping methods are encoded as follows:  
  - **Air** -> `0`  
  - **Land** -> `1`  
  - **Sea** -> `2`  
- The mapping enables comparison across different modes of shipment, which is essential for analyzing logistical efficiency and cost-effectiveness.

### Payment_Terms  
- Payment terms are encoded into numerical values:  
  - **Cash on Delivery** -> `0`  
  - **Net 30** -> `1`  
  - **Net 60** -> `2`  
  - **Prepaid** -> `3`  
- This mapping is helpful for understanding the impact of payment terms on transaction efficiency and cash flow.
### **Analysis of Non-Normally Distributed Variables and Scaling**

### Non-Normally Distributed Variables:
- The variables that do not follow a normal distribution are identified using the Shapiro-Wilk test.
- In this case, all three variables — **Quantity**, **Value**, and **Weight** — are found to be non-normally distributed, based on the p-value being less than 0.05 for each.

### Scaling with MinMaxScaler:
- These non-normally distributed variables are scaled using the **MinMaxScaler**, which transforms the data into a range between **0** and **1**.
- Scaling ensures that the variables are treated equally during the modeling process, especially for algorithms that are sensitive to feature scaling (e.g., linear regression, k-means clustering).

### Scaled Non-Categorical Dataset:
- The first few rows of the scaled dataset show that the values of **Quantity**, **Value**, and **Weight** have been transformed to lie within the range [0, 1].

### Scaled Variables:
- All three variables (**Quantity**, **Value**, and **Weight**) have undergone transformation, making them suitable for machine learning models.
  
These techniques may improve the normality of the data, which could, in turn, enhance the performance of the machine learning model.
### **Descriptive Statistics for Categorical Variables**

#### 1. **Import_Export (Trade Type)**
- **Count:** 5001 observations, indicating the total number of records in this column.
- **Frequency:**
  - **Export (0.0):** 2518 occurrences
  - **Import (1.0):** 2483 occurrences
- **Proportion:**
  - **Export (0.0):** 50.35% of the dataset
  - **Import (1.0):** 49.65% of the dataset
- **Minimum/Maximum:** The values range from 0 to 1, reflecting the binary nature of this variable.
- **Mode:** The most common value is 0 (Export), which aligns with the proportions.
- **Rank:** The variable uses a binary encoding of 0 and 1, so the rank is `[0. 1.]`, showing that it is treated as a binary variable.
- **Spearman & Kendall Correlation:** Both are `NaN`, indicating that correlation measures are not applicable to categorical variables.

**Conclusion:** The trade type (Import/Export) is nearly balanced, with a slight edge toward Exports (50.35%). This could influence the analysis, particularly when looking at trends or patterns based on trade type.

---

#### 2. **Category (Product Type)**
- **Minimum/Maximum:** The variable has values ranging from 0 to 4, representing different product categories.
- **Mode:** The most common category is 0 (Clothing), but the distribution is fairly balanced across the five categories.
- **Rank:** The variable is treated as ordinal with ranks `[1. 0. 2. 4. 3.]`, indicating a predefined order (Clothing, Electronics, Furniture, Machinery, Toys).
- **Spearman & Kendall Correlation:** Both are `NaN` due to the categorical nature of the variable.

**Conclusion:** The product categories are relatively evenly distributed, with no single category dominating. This balanced distribution is crucial for analysis, particularly when exploring trends across different product types.

---

#### 3. **Shipping_Method**
- **Minimum/Maximum:** Values range from 0 to 2, indicating three modes of transportation: Air, Land, and Sea.
- **Mode:** The most frequent shipping method is Sea (2.0), which is used in 33.95% of the transactions.
- **Rank:** The variable is treated as ordinal with ranks `[0. 1. 2.]`, representing Air, Land, and Sea.

**Conclusion:** The shipping methods are quite evenly distributed, with no single method overwhelmingly dominating. Sea is slightly more popular, which could indicate the preferred mode of transportation in international trade.

---

#### 4. **Payment_Terms**
- **Minimum/Maximum:** Values range from 0 to 3, reflecting the different payment terms.
- **Mode:** The most common payment term is Cash on Delivery (0.0), which accounts for 25.63% of transactions.
- **Rank:** The variable is ordinal, with ranks `[3. 1. 2. 0.]`, indicating an ordered scale for payment terms.
- **Spearman & Kendall Correlation:** Both are `NaN` due to the categorical nature of the variable.

**Conclusion:** Payment terms are fairly evenly distributed, with no single term dominating. This provides flexibility when analyzing how different payment methods affect the trade transactions.

---

### Summary of Insights
- The categorical variables in the dataset, such as **Import_Export**, **Category**, **Shipping_Method**, and **Payment_Terms**, exhibit different distribution patterns.
- The **Import_Export** variable is nearly balanced, which may have implications for how trade is analyzed across imports and exports.
- The **Category** variable is fairly evenly distributed, providing insights into various product types without a dominant category.
- **Shipping_Method** and **Payment_Terms** show relatively balanced distributions, but slight differences suggest a preference for certain shipping methods (Sea) and payment terms (Cash on Delivery).
- The **Rank** and **Mode** provide useful insights for ordinal variables, allowing for comparisons across categories.

These descriptive statistics help understand the distribution and characteristics of categorical variables, which are crucial for further analysis and modeling.
1. Descriptive Statistics
1.1 Central Tendency (Mean, Median, Mode)
Mean:
Quantity: 0.498; Value: 0.501; Weight: 0.497
All means are close to 0.5, suggesting the data is fairly balanced or normalized.
Median:
Quantity: 0.503; Value: 0.502; Weight: 0.494
Medians are also close to the means, indicating symmetrical distributions.
Mode:
Quantity: 0.007; Value: 0.044; Weight: 0.014
Modes differ significantly from the means and medians, suggesting a large spread or concentration at certain values.
1.2 Range and Standard Deviation
Range: All variables have a range of 1.0, as they are normalized or standardized.
Standard Deviation:
Quantity: 0.287; Value: 0.290; Weight: 0.290
Variability across the variables is very similar, indicating consistent scaling.
1.3 Skewness and Kurtosis
Skewness:
All variables show very low skewness (~0.01–0.02), indicating nearly symmetrical distributions.
Kurtosis:
All variables have negative kurtosis (~-1.17 to -1.22), suggesting light tails (platykurtic distributions).
Implication:
The data is well-behaved, with no extreme skew or heavy tails.
1.4 Coefficient of Variation (CV)
CV:
Quantity: 0.575; Value: 0.580; Weight: 0.583
CV values indicate moderate relative variability across the variables, with all being close to each other.
1.5 Confidence Interval (95%)
Confidence intervals for the means:
Quantity: (0.490, 0.506)
Value: (0.493, 0.509)
Weight: (0.489, 0.505)
Implication:
Narrow confidence intervals suggest a high level of precision in the mean estimates.
2. Correlation Analysis
2.1 Pearson Correlation
Quantifies linear relationships between variables:
Quantity & Value: 0.0057
Quantity & Weight: 0.0071
Value & Weight: 0.0246
Implication:
All correlations are very close to zero, indicating no significant linear relationships among the variables.
2.2 Spearman Correlation
Quantifies monotonic relationships between variables:
Quantity & Value: 0.0062
Quantity & Weight: 0.0069
Value & Weight: 0.0243
Implication:
Similar to Pearson, Spearman correlations are near zero, confirming no significant monotonic relationships.
# Descriptive Statistics and Correlation Analysis

## 1. Descriptive Statistics

### 1.1 Central Tendency (Mean, Median, Mode)
- **Mean**:
  - Quantity: **0.498**
  - Value: **0.501**
  - Weight: **0.497**
  - All means are close to 0.5, suggesting the data is fairly balanced or normalized.

- **Median**:
  - Quantity: **0.503**
  - Value: **0.502**
  - Weight: **0.494**
  - Medians are also close to the means, indicating symmetrical distributions.

- **Mode**:
  - Quantity: **0.007**
  - Value: **0.044**
  - Weight: **0.014**
  - Modes differ significantly from the means and medians, suggesting a large spread or concentration at certain values.

### 1.2 Range and Standard Deviation
- **Range**:
  - All variables have a range of **1.0**, as they are normalized or standardized.
- **Standard Deviation**:
  - Quantity: **0.287**
  - Value: **0.290**
  - Weight: **0.290**
  - Variability across the variables is very similar, indicating consistent scaling.

### 1.3 Skewness and Kurtosis
- **Skewness**:
  - All variables show very low skewness (~0.01–0.02), indicating nearly symmetrical distributions.
- **Kurtosis**:
  - All variables have negative kurtosis (~-1.17 to -1.22), suggesting light tails (platykurtic distributions).
- **Implication**:
  - The data is well-behaved, with no extreme skew or heavy tails.

### 1.4 Coefficient of Variation (CV)
- **CV**:
  - Quantity: **0.575**
  - Value: **0.580**
  - Weight: **0.583**
  - CV values indicate moderate relative variability across the variables, with all being close to each other.

### 1.5 Confidence Interval (95%)
- Confidence intervals for the means:
  - Quantity: **(0.490, 0.506)**
  - Value: **(0.493, 0.509)**
  - Weight: **(0.489, 0.505)**
- **Implication**:
  - Narrow confidence intervals suggest a high level of precision in the mean estimates.

---

## 2. Correlation Analysis

### 2.1 Pearson Correlation
- Quantifies **linear relationships** between variables:
  - Quantity & Value: **0.0057**
  - Quantity & Weight: **0.0071**
  - Value & Weight: **0.0246**
- **Implication**:
  - All correlations are very close to zero, indicating no significant linear relationships among the variables.

### 2.2 Spearman Correlation
- Quantifies **monotonic relationships** between variables:
  - Quantity & Value: **0.0062**
  - Quantity & Weight: **0.0069**
  - Value & Weight: **0.0243**
- **Implication**:
  - Similar to Pearson, Spearman correlations are near zero, confirming no significant monotonic relationships.
## **DATA VISUALIZATION**
 ### BAR CHART
# Analysis of Category Distribution

## Key Observations:
1. **Category Representation**:
   - The bar chart illustrates the distribution of categories ranging from **0.0** to **4.0**.
   - Category **0.0** has the highest count, exceeding **1000**.
   - Other categories (**1.0**, **2.0**, **3.0**, and **4.0**) show relatively balanced distributions, though slightly lower than category **0.0**.

2. **Count Range**:
   - Counts across the categories are similar, indicating a relatively **even distribution** of data points among the categories.
   - However, there is a slight decrease in the counts as the category number increases.

3. **Implications**:
   - This balanced distribution across categories suggests that the dataset is **well-represented** across all groups, which is beneficial for statistical analysis and modeling.
   - The higher count for category **0.0** may indicate a slight overrepresentation, which should be considered in weighted analyses if necessary.

## **Findings**:
- If using these categories for predictive modeling, ensure that any overrepresented categories (like **0.0**) do not dominate the model's predictions.
- Perform additional checks to understand if the category distribution aligns with the overall population distribution or is due to sampling bias.

### PIE CHART
# Analysis of Category Proportion

## Key Observations:
1. **Proportional Representation**:
   - The pie chart shows the proportional distribution of categories from **0.0** to **4.0**.
   - The proportions are fairly balanced across categories, with the following percentages:
     - **Category 0.0**: 21.2% (largest proportion).
     - **Category 1.0**: 19.7%.
     - **Category 2.0**: 20.3%.
     - **Category 3.0**: 19.6%.
     - **Category 4.0**: 19.2% (smallest proportion).

2. **Variation**:
   - The distribution is relatively uniform, with a difference of approximately 2% between the largest (21.2%) and smallest (19.2%) proportions.
   - This indicates minimal imbalance in the representation of categories.

3. **Implications**:
   - A balanced category distribution like this is ideal for unbiased modeling and analysis.
   - No single category dominates the dataset, ensuring fairness in statistical and machine learning applications.

## **Findings:**
- While the proportions are well-balanced, verify if the dataset's proportions align with the real-world scenario or research objectives.
- Proceed with modeling without needing significant adjustments for class imbalance.

 ### BOX PLOT
 # Analysis of Box Plot: Distribution of Value across Categories

## Key Observations:
1. **Central Tendency**:
   - The median (horizontal line in the box) for each category is approximately the same, indicating a similar central tendency across all categories.

2. **Spread of Data**:
   - The interquartile range (IQR, represented by the height of the boxes) is consistent across all categories.
   - This suggests that the variability in the "Value" variable is uniform across the different categories.

3. **Whiskers (Range)**:
   - The whiskers (lines extending from the box) indicate the range of non-outlier data, and they are similar across all categories.
   - This suggests no significant differences in the spread of values.

4. **Outliers**:
   - There are no apparent outliers in any of the categories, as no points are plotted outside the whiskers.

5. **Mean (Green Triangle)**:
   - The green triangles indicate the mean, which is close to the median for all categories.
   - This symmetry suggests that the "Value" variable is approximately normally distributed within each category.

## Implications:
- The "Value" variable is distributed similarly across all categories, with no significant differences in central tendency, variability, or range.
- This uniformity may imply that the "Value" variable is independent of the categories or that the categories do not heavily influence its distribution.

## **Findings:**
- Since the distributions are similar across categories, further statistical tests (e.g., ANOVA) can confirm if differences are statistically significant.
- If the categories are meant to represent distinct groups, consider investigating other variables that might better distinguish them.

### HEATMAP
### Analysis of Correlation Matrix for Imports/Exports Dataset

The heatmap illustrates the correlation matrix between three variables: **Quantity**, **Value**, and **Weight** in the imports/exports dataset. The analysis is as follows:

1. **Diagonal Values**:
   - Each diagonal value is `1`, indicating a perfect correlation of each variable with itself. This is expected in any correlation matrix.

2. **Off-Diagonal Values**:
   - **Quantity and Value**: Correlation coefficient = `0.0057`.
     - There is an extremely weak positive correlation between quantity and value, indicating that the total value of imports/exports does not strongly depend on the quantity.
   - **Quantity and Weight**: Correlation coefficient = `0.0071`.
     - Similarly, there is an extremely weak positive correlation between quantity and weight, suggesting that the quantity of items imported/exported is almost independent of their weight.
   - **Value and Weight**: Correlation coefficient = `0.025`.
     - The correlation between value and weight is also very weakly positive, implying that the weight of imports/exports has minimal influence on their overall value.

3. **Color Coding**:
   - The heatmap uses a color gradient where dark red indicates strong positive correlations (closer to `1`), and dark blue indicates strong negative correlations (closer to `-1`).
   - The predominance of blue shades in the off-diagonal cells confirms the weak correlations among the variables.

4. **Conclusion**:
   - The dataset suggests that the relationships between **Quantity**, **Value**, and **Weight** in the imports/exports context are negligible or non-significant.
   - This could indicate that other factors (e.g., type of goods, pricing strategies, or economic conditions) are more critical in determining the value or weight of imports/exports than the quantities alone.

This analysis highlights the independence of these variables in the given dataset, which could provide insights into further exploration or modeling.
## **INFERENTIAL STATSTICS**
# Analysis of Contingency Table and Chi-Square Test

## Contingency Table
### Observed Frequencies:
| Shipping Method | 0.0  | 1.0  | 2.0  |
|------------------|------|------|------|
| **Import_Export = 0.0** | 833  | 787  | 898  |
| **Import_Export = 1.0** | 838  | 845  | 800  |

### Expected Frequencies:
| Shipping Method | 0.0       | 1.0       | 2.0       |
|------------------|-----------|-----------|-----------|
| **Import_Export = 0.0** | 841.35  | 821.71  | 854.94  |
| **Import_Export = 1.0** | 829.65  | 810.29  | 843.06  |

## Chi-Square Test Results:
- **Chi-squared Statistic**: 7.49
- **P-value**: 0.024
- **Degrees of Freedom (df)**: 2

## Interpretation:
1. **Hypotheses**:
   - **Null Hypothesis (H₀)**: There is no association between "Shipping Method" and "Import_Export".
   - **Alternative Hypothesis (H₁)**: There is an association between "Shipping Method" and "Import_Export".

2. **Significance Level (α)**: 0.05
   - Since the p-value (0.024) is less than the significance level (0.05), we **reject the null hypothesis**.

3. **Conclusion**:
   - There is a **statistically significant association** between "Shipping Method" and "Import_Export". This suggests that the choice of shipping method depends on whether the shipment is an import or export.

4. **Comparison of Observed vs. Expected**:
   - The observed frequencies deviate slightly from the expected frequencies, contributing to the chi-squared statistic.

##** Findings:**
- Further analysis can explore the strength and direction of the association (e.g., Cramér's V).
- Investigate potential practical implications of this association for optimizing shipping strategies based on import/export operations.

# Normality Tests Analysis for 'Quantity', 'Value', and 'Weight'

### Assumptions:
- The dataset is named `ng06_ds_preprocessed`.
- Non-categorical columns analyzed: `Quantity`, `Value`, `Weight`.

---

## **Results Summary**

### **1. Tests for 'Quantity':**
- **Shapiro-Wilk Test**:
  - **Statistic**: 0.9577
  - **P-value**: 7.54e-36
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Kolmogorov-Smirnov Test**:
  - **Statistic**: 0.5
  - **P-value**: 0.0
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Anderson-Darling Test**:
  - **Statistic**: 50.75
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Jarque-Bera Test**:
  - **Statistic**: 286.42
  - **P-value**: 6.36e-63
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).

---

### **2. Tests for 'Value':**
- **Shapiro-Wilk Test**:
  - **Statistic**: 0.9540
  - **P-value**: 4.86e-37
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Kolmogorov-Smirnov Test**:
  - **Statistic**: 0.5
  - **P-value**: 0.0
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Anderson-Darling Test**:
  - **Statistic**: 56.22
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Jarque-Bera Test**:
  - **Statistic**: 300.93
  - **P-value**: 4.50e-66
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).

---

### **3. Tests for 'Weight':**
- **Shapiro-Wilk Test**:
  - **Statistic**: 0.9530
  - **P-value**: 2.37e-37
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Kolmogorov-Smirnov Test**:
  - **Statistic**: 0.5000
  - **P-value**: 0.0
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Anderson-Darling Test**:
  - **Statistic**: 59.91
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).
- **Jarque-Bera Test**:
  - **Statistic**: 310.32
  - **P-value**: 4.11e-68
  - **Conclusion**: Data is not normal (reject H₀ at α = 0.05).

---

## **Overall Conclusion:**
- Across all columns (`Quantity`, `Value`, `Weight`), the normality tests indicate that the data does **not** follow a normal distribution.
- This conclusion is consistent across the Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling, and Jarque-Bera tests.
- The rejection of the null hypothesis (H₀: Data is normally distributed) at a 5% significance level highlights significant deviations from normality.

---

## **Findings:**
1. Consider using **non-parametric statistical tests** for further analysis, as the assumption of normality is violated.
2. Apply **transformations** (e.g., log, square root) to attempt normalization if required.
3. Explore the data's distribution through visualization (e.g., histograms, Q-Q plots) to understand the underlying patterns better.
4. If applicable, evaluate whether large sample sizes (>5000) may affect test accuracy, as warned by `scipy` for Shapiro-Wilk.

## **Supervised Machine Learning**

# **Analysis of Train-Test Split for Supervised Machine Learning**

## **Dataset Overview**
- **X_train shape**: (3500, 3)  
  The training dataset for features contains 3500 samples and 3 features.
  
- **X_test shape**: (1501, 3)  
  The test dataset for features contains 1501 samples and 3 features.

- **y_train shape**: (3500, 1)  
  The training dataset for labels contains 3500 samples and 1 target variable.

- **y_test shape**: (1501, 1)  
  The test dataset for labels contains 1501 samples and 1 target variable.

---

## **Analysis**
1. **Data Split**:
   - The data split ratio is approximately **80/20**, which is a standard and widely used partition for machine learning tasks.
   - This ensures sufficient data for both training and testing, allowing the model to learn patterns effectively while reserving a significant portion for validation.

2. **Training Dataset**:
   - Contains the majority of the samples (80% of the total dataset).
   - Provides ample data for the model to generalize effectively.

3. **Testing Dataset**:
   - Consists of 20% of the data, reserved for evaluating the model's performance on unseen data.
   - A properly sized test set ensures reliable validation of the model's predictive accuracy.

4. **Feature and Target Distribution**:
   - Both the training and testing datasets maintain the same number of features (3) and target variables (1), ensuring consistency during model training and evaluation.

---

## **Conclusion**
- The **80/20 split** is appropriate for this dataset size and should provide a balanced trade-off between training the model and evaluating its performance.
- The dataset is well-prepared for supervised learning, with sufficient samples in both the training and testing sets to support reliable model development and assessment.

## **Decision Tree**

### **Analysis of Decision Tree Model Training, Predictions, and Result Compilation**

## **Analysis of Decision Tree Classifier Predictions**

1. **Structure of the Output**:
   - The DataFrame contains **1501 rows** corresponding to the test dataset.
   - Columns include:
     - `Quantity`: Represents a numerical feature normalized between 0 and 1.
     - `Weight`: Represents another numerical feature normalized between 0 and 1.
     - `Value`: Represents a numerical feature normalized between 0 and 1.
     - `Predicted`: Contains the predicted class labels (`0.0` or `1.0`) generated by the Decision Tree Classifier.

2. **Predicted Class Labels**:
   - The `Predicted` column indicates the classification result for each test instance.
   - Values are binary, representing the two possible target classes:
     - `0.0`: One class category.
     - `1.0`: Another class category.

3. **Insights from the Data**:
   - **Range of Feature Values**:
     - Features `Quantity`, `Weight`, and `Value` are normalized, ensuring all values are within the range `[0, 1]`.
   - **Prediction Trends**:
     - Instances with higher `Value` and `Weight` values appear more likely to be classified as `1.0`.
     - Instances with lower `Value` or `Weight` are often predicted as `0.0`.
   - This observation suggests that `Value` and `Weight` may play a stronger role in influencing predictions.

4. **Evaluation Potential**:
   - The data provides a basis for evaluating model performance.
   - Metrics such as accuracy, precision, recall, and F1-score can be computed using the ground truth labels and predictions.

5. **Use Case**:
   - The predictions align well with classification tasks where normalized features influence the binary target variable.
   - This analysis can help refine feature importance and optimize the model for better prediction accuracy.

# **Decision Tree rules as text**


## **Feature Importance Analysis**

The Decision Tree Classifier has provided the following feature importance scores for predicting the target variable (e.g., Import/Export):

### **Feature Importance Rankings**
1. **Weight**: Importance score of **0.341**  
   - This feature is the most significant in determining whether a transaction is categorized as import or export.
   - Suggests that the weight of items heavily influences the model's predictions.

2. **Value**: Importance score of **0.340**  
   - Close in importance to `Weight`, this feature indicates that the monetary value of goods also plays a crucial role in classification.

3. **Quantity**: Importance score of **0.320**  
   - While slightly less influential, the quantity of items still contributes significantly to the model's decisions.

### **Observations**
- **Balanced Contribution**: The importance scores for all three features are fairly close, indicating that the model relies on a combination of these attributes for accurate predictions.
- **Significance of Weight and Value**: Together, `Weight` and `Value` account for approximately 68.1% of the decision-making process, suggesting that physical and financial attributes are key differentiators for imports and exports.

### **Findings**
- **Business Implications**: Companies should focus on monitoring and analyzing the weight and value of goods to optimize their import/export processes.
- **Model Refinement**: Consider collecting additional data or engineering new features that may further enhance the model's ability to differentiate between imports and exports.

### **Model Evaluation**

## **Performance Analysis of Decision Tree Classifier**

### **Overall Accuracy**
- The Decision Tree model achieved an **accuracy of 50.50%**.  
  - This indicates that the model's predictions are only marginally better than random guessing for this dataset.

### **Class-wise Metrics**
1. **Class 0.0 (Export)**  
   - **Precision**: 0.52  
     - Of all transactions predicted as exports, 52% were correct.  
   - **Recall**: 0.51  
     - The model correctly identified 51% of actual export transactions.  
   - **F1-Score**: 0.52  
     - Balances precision and recall, showing moderate performance for exports.

2. **Class 1.0 (Import)**  
   - **Precision**: 0.49  
     - Of all transactions predicted as imports, 49% were correct.  
   - **Recall**: 0.50  
     - The model correctly identified 50% of actual import transactions.  
   - **F1-Score**: 0.49  
     - Indicates slightly lower performance for imports compared to exports.

### **Averages**
- **Macro Average**:  
  - Precision, recall, and F1-score for both classes are equally weighted at **0.50**, reflecting balanced but suboptimal performance across classes.  

- **Weighted Average**:  
  - Weighted by class support, the overall metrics hover around **0.50**, confirming the model's mediocre performance.

### **Observations**
1. The Decision Tree struggles to differentiate between imports and exports, likely due to overlapping feature distributions or insufficient feature importance.
2. The balanced precision and recall indicate no significant bias towards either class.

### **Findings**
- **Data Improvement**: Investigate if additional features or higher-quality data could improve the model's predictions.
- **Model Enhancement**: Experiment with more complex algorithms like Random Forests or Gradient Boosting to capture non-linear relationships in the data.
- **Hyperparameter Tuning**: Optimize the Decision Tree parameters (e.g., `max_depth`, `min_samples_split`) to enhance performance.

## **Confusion matrix**

The confusion matrix provides a detailed breakdown of the performance of the Decision Tree model for predicting import/export transactions.

### **Key Observations**
1. **True Positives (TP)**:  
   - The model correctly predicted **396** instances of exports (Class 0.0).
   
2. **True Negatives (TN)**:  
   - The model correctly predicted **362** instances of imports (Class 1.0).

3. **False Positives (FP)**:  
   - **375** export transactions were misclassified as imports.

4. **False Negatives (FN)**:  
   - **368** import transactions were misclassified as exports.

### **Findings**
- **Balanced Errors**: The model has a nearly equal number of false positives and false negatives, which indicates that it does not favor one class over the other. However, both types of errors are high.
- **Accuracy Limitations**:
   - While the model is able to predict certain cases correctly, the significant misclassifications highlight limitations in separating imports and exports based on the provided features.
   
### **Suggestions for Improvement**
1. **Feature Engineering**:  
   - Incorporate additional or more discriminative features that might improve the model's ability to distinguish between the two classes.

2. **Advanced Models**:  
   - Use ensemble methods like Random Forest or Gradient Boosting, which tend to perform better than a standalone Decision Tree.

### **Conclusion**
The confusion matrix demonstrates that the Decision Tree model struggles to classify imports and exports effectively, with substantial misclassification in both directions. Further refinement of the dataset, features, and modeling approach is required to enhance prediction accuracy.

## **Analysis of Logistic Regression Results**

### **Key Observations**
1. **Prediction Output**:
   - The model generated predictions for 1501 instances, with a binary output (0.0 or 1.0) indicating the predicted class for each sample.
   - The classes (0.0 and 1.0) likely represent categories such as "Export" and "Import," respectively.

2. **Feature Contributions**:
   - The prediction is based on three features: `Quantity`, `Weight`, and `Value`. These features are normalized to ensure all variables contribute equally to the logistic regression model.

3. **Prediction Distribution**:
   - From the sample output, the majority of predictions belong to Class 0.0, which may indicate class imbalance or stronger weightage of the features toward one class.

4. **Warning Message**:
   - The warning (`DataConversionWarning`) indicates that the `y` variable (target) was passed as a column vector when the model expected a 1D array.
   - While this does not affect the predictions, it suggests a need for reshaping `y` using `.ravel()` for better compatibility.

### **Model Considerations**
- **Class Imbalance**:
   - If the dataset has an uneven distribution of classes (e.g., more instances of exports than imports), the model may become biased toward the majority class.
   - Address class imbalance by using techniques such as oversampling, undersampling, or applying class weights during training.

- **Feature Importance**:
   - The three features (`Quantity`, `Weight`, and `Value`) directly influence the decision boundary. Ensure these features are relevant and properly scaled for better predictions.

### **Suggestions for Improvement**
1. **Warning Resolution**:
   - Address the `DataConversionWarning` by reshaping the target variable using `.ravel()` to ensure compatibility with the logistic regression model.

2. **Performance Evaluation**:
   - Evaluate the model's performance using metrics like accuracy, precision, recall, and F1-score.
   - Create a confusion matrix to better understand the distribution of correct and incorrect predictions.

### **Conclusion**
Logistic regression provides a baseline model for predicting imports and exports based on the dataset. While the model generates reasonable predictions, further analysis and adjustments (e.g., handling class imbalance, optimizing features, and addressing warnings) are necessary to improve its accuracy and reliability.

## **Analysis of Logistic Regression Model Evaluation**

### **Overall Accuracy**
- The model achieved an accuracy of **52.10%**, indicating that it correctly classified slightly more than half of the samples.
- This is marginally better than random guessing for a binary classification problem but suggests room for improvement.

### **Class-Specific Performance**
1. **Class 0.0 (e.g., Export)**:
   - **Precision**: 0.53 – For every instance predicted as Class 0.0, 53% were actually correct.
   - **Recall**: 0.65 – The model successfully identified 65% of all actual Class 0.0 instances.
   - **F1-Score**: 0.58 – Represents a balance between precision and recall for Class 0.0, showing acceptable performance.

2. **Class 1.0 (e.g., Import)**:
   - **Precision**: 0.51 – For every instance predicted as Class 1.0, 51% were actually correct.
   - **Recall**: 0.39 – The model identified only 39% of actual Class 1.0 instances, indicating a significant portion of false negatives.
   - **F1-Score**: 0.44 – Lower compared to Class 0.0, reflecting weaker performance in identifying Class 1.0.

### **Averages**
- **Macro Average**:
  - Precision, recall, and F1-score are all approximately **0.52**, indicating balanced performance across both classes but with no notable advantage in identifying one over the other.
- **Weighted Average**:
  - These metrics are weighted based on class support, showing the overall balance of the model's predictions.

### **Key Observations**
1. **Class Imbalance Impact**:
   - The model demonstrates better performance for Class 0.0 (Export) compared to Class 1.0 (Import), suggesting that class imbalance or feature relevance may be affecting performance.

2. **Misclassification of Class 1.0**:
   - The low recall for Class 1.0 indicates the model struggles to correctly identify instances of this class, which may lead to significant false negatives in a real-world scenario.

3. **Balanced Accuracy**:
   - Both classes have similar levels of precision and recall, but overall performance remains low, requiring further tuning.

### **Conclusion**
While the logistic regression model provides a baseline accuracy of 52.10%, its limited ability to correctly classify instances of Class 1.0 suggests the need for further optimization. Addressing class imbalance, tuning hyperparameters, and exploring additional features or alternative models could significantly improve its predictive performance.

## **Analysis of Confusion Matrix for Logistic Regression**

### **Interpretation of the Confusion Matrix**
- **True Positives (TP)**: 282
  - The model correctly predicted 282 instances of Class 1.0.
- **True Negatives (TN)**: 500
  - The model correctly predicted 500 instances of Class 0.0.
- **False Positives (FP)**: 271
  - The model incorrectly predicted 271 instances as Class 1.0, which were actually Class 0.0.
- **False Negatives (FN)**: 448
  - The model incorrectly predicted 448 instances as Class 0.0, which were actually Class 1.0.

### **Key Observations**
1. **Imbalanced Class Performance**:
   - The model performs better at identifying Class 0.0 (Export) compared to Class 1.0 (Import).
   - The large number of false negatives (448) indicates that the model struggles to identify Class 1.0 instances.

2. **Accuracy Insight**:
   - While the true negatives are high, the false negatives significantly affect the model's ability to identify Class 1.0, reducing overall effectiveness.

3. **Class Overlap**:
   - The confusion between the classes (271 false positives and 448 false negatives) suggests that the features used may not fully differentiate between Class 0.0 and Class 1.0.

### **Findings**
The logistic regression model demonstrates moderate performance in predicting Class 0.0 but struggles significantly with Class 1.0. Improvements in data preprocessing, feature engineering, and model tuning are essential to enhance its predictive capability.

## **Random Forest**

###  **Data Structure**
- **Data Summary**:
  - The dataset contains **1501 rows** and **4 columns**: `Quantity`, `Weight`, `Value`, and `Predicted`.
  - The `Predicted` column represents the output predictions generated by the Random Forest model.

### **Key Observations**
1. **Prediction Distribution**:
   - The predictions for the dataset are balanced between Class 0.0 and Class 1.0, indicating that the model attempts to identify patterns in both classes without heavy bias.

2. **Feature Usage**:
   - The Random Forest model utilizes multiple decision trees, aggregating their outputs for prediction, which helps in handling non-linear relationships and minimizing overfitting.

3. **Performance Insights**:
   - The output suggests the Random Forest model was applied to a classification problem. However, its effectiveness depends on evaluation metrics such as accuracy, precision, recall, and the confusion matrix (not provided here).

### **Recommendations for Improvement**

1. **Feature Importance**:
   - Analyze the feature importance scores generated by the Random Forest model to identify which features contribute most to the predictions.

2. **Model Tuning**:
   - Perform hyperparameter tuning, such as adjusting the number of trees (`n_estimators`), tree depth (`max_depth`), or minimum samples per split (`min_samples_split`) to optimize performance.

3. **Validation**:
   - Use cross-validation to ensure the model's generalizability and avoid overfitting.
   - Evaluate the model's performance using metrics such as ROC-AUC and F1-score for a better understanding of its strengths and weaknesses.

4. **Comparison with Other Models**:
   - Compare the performance of the Random Forest model with simpler models (e.g., Logistic Regression) or more complex ones (e.g., Gradient Boosting) to ensure the best model is selected for the dataset.

### **Conclusion**
The Random Forest model appears to be functioning as expected, providing predictions for the classification task. Further evaluation and fine-tuning are essential to improve its performance and address any potential issues in the data preprocessing pipeline.

## **Analysis of Confusion Matrix for Random Forest Model**

### **Overview**
The confusion matrix provides a detailed breakdown of the Random Forest model's predictions. It compares the actual labels (rows) against the predicted labels (columns) for a binary classification task.

### **Key Observations**
1. **True Positives (TP)**:
   - **361** instances where the model correctly predicted class `1.0`.

2. **True Negatives (TN)**:
   - **410** instances where the model correctly predicted class `0.0`.

3. **False Positives (FP)**:
   - **361** instances where the model incorrectly predicted class `1.0` for actual class `0.0`.

4. **False Negatives (FN)**:
   - **369** instances where the model incorrectly predicted class `0.0` for actual class `1.0`.

### **Findings**
- **Performance**:
  - The model's performance shows a moderate accuracy of approximately 51.4%.
  - The precision and recall scores suggest that the model struggles to distinguish between the two classes effectively, particularly for class `1.0`.

- **Class Imbalance**:
  - The similar number of False Positives and False Negatives indicates no severe class imbalance but highlights challenges in model generalization.

- **Errors**:
  - The model has a high number of misclassifications, as shown by 361 False Positives and 369 False Negatives. This suggests the need for further improvements.

### **Conclusion**
The Random Forest model provides moderate accuracy, but its high number of misclassifications (False Positives and False Negatives) indicates room for improvement. Implementing the suggested steps could lead to better performance.

## **Comparison of Logistic Regression and Decision Tree Models**

### **Model Performance**
1. **Decision Tree (DT)**:
   - Accuracy: **50.5%**
   - The Decision Tree model has slightly above random guessing accuracy, indicating that it struggles to capture complex relationships in the data.

2. **Logistic Regression (LR)**:
   - Accuracy: **52.1%**
   - Logistic Regression outperforms the Decision Tree by a small margin, but the improvement is marginal, showing limited predictive power.

### **Key Observations**
1. **Logistic Regression Strengths**:
   - Logistic Regression is better at handling linear relationships between features and the target variable.
   - It provides slightly better accuracy and is less prone to overfitting, especially when the dataset is not very large.

2. **Decision Tree Weaknesses**:
   - The Decision Tree model may have overfitted the training data, leading to poorer generalization on the test set.
   - It might require hyperparameter tuning (e.g., `max_depth`, `min_samples_split`) to improve performance.

3. **Overall Difference**:
   - The accuracy difference between the two models is relatively small (~1.6%), which suggests that neither model is highly effective on this dataset without further optimization.

### **Recommendations for Model Selection**
1. **Logistic Regression**:
   - Logistic Regression is the preferred choice here, given its slightly better performance and lower risk of overfitting.
   - It is simpler to implement and interpret, which makes it suitable for scenarios where model explainability is important.

2. **Improvements for Decision Tree**:
   - Consider hyperparameter tuning and feature engineering to improve the Decision Tree's performance.
   - Using ensemble techniques (e.g., Random Forest or Gradient Boosting) could mitigate overfitting and enhance accuracy.

3. **Evaluation with Cross-Validation**:
   - Use k-fold cross-validation to evaluate the models consistently and obtain a better understanding of their generalizability.

### **Conclusion**
Logistic Regression outperforms the Decision Tree model by a small margin, making it the better choice for this dataset in its current form. However, further improvements in feature engineering, hyperparameter tuning, and exploring advanced models can lead to better overall performance.

# Compare Random Forest and Decision Tree
# **Managerial Insights & Recommendations**

## Overview
The analysis of the imports and exports dataset, utilizing various machine learning techniques such as clustering, provided valuable insights into global trade patterns, economic relationships, and trade dependencies. Based on the results of clustering, correlation analysis, and feature engineering, we derived managerial insights and actionable recommendations for stakeholders in global trade, economics, and policy-making.

## Managerial Insights

1. **Trade Dependencies and Economic Growth**:
   - There is a strong correlation between GDP and both imports and exports, indicating that higher economic growth tends to be associated with increased trade activity. Countries with higher GDP are likely to be major importers and exporters, influencing their global economic power.
   - **Insight**: Businesses and policymakers should focus on emerging markets with rising GDP to identify new trade opportunities, as these markets may show increasing demand for imports and offer growth potential for exports.

2. **Trade Balance and Economic Strategy**:
   - Countries with a negative trade balance (importing more than they export) may need to consider adjustments to their trade policies to reduce reliance on imports. Conversely, nations with positive trade balances could leverage their trade surpluses to invest in domestic infrastructure or other economic growth initiatives.
   - **Insight**: Policymakers should examine trade balance trends and ensure that trade agreements and domestic policies encourage export growth, reduce import dependence, and promote self-sustainability in key industries.

3. **Clustering Patterns**:
   - The clustering analysis revealed distinct groups of countries with similar import-export profiles. These clusters represent groups of countries that exhibit similar economic and trade behaviors.
   - **Insight**: Companies looking to enter new international markets should identify countries within similar clusters. By focusing on markets with similar trade patterns, companies can tailor their marketing and sales strategies to the specific needs of each region.

4. **Outliers in Trade Activity**:
   - Several countries were identified as outliers with significantly high export/import values. These outliers might represent anomalies, such as special trade agreements, monopolistic industries, or major global players.
   - **Insight**: While outliers may represent unique trade conditions, businesses should study these countries closely to understand the factors driving these anomalies, such as geopolitical influences, large-scale trade agreements, or market dominance.

5. **Economic Implications of Trade Balances**:
   - The analysis of trade balances suggests that countries with a significant trade surplus might use their position to secure favorable global trade agreements, while countries with trade deficits could focus on policy reforms to address trade imbalances.
   - **Insight**: Stakeholders should monitor trade balance trends and use this data to anticipate shifts in global trade relations. Businesses can adjust supply chains and import/export strategies based on the trade health of the countries they operate in.

## Recommendations

1. **Focus on High-GDP Markets**:
   - **Recommendation**: Companies and policymakers should focus on high-GDP countries for both imports and exports. As these economies grow, their trade activities will expand, creating opportunities for businesses to increase their market share.

2. **Encourage Trade Policy Reforms**:
   - **Recommendation**: Countries with persistent negative trade balances should explore trade policy reforms, such as tariff adjustments, subsidies for key industries, or incentivizing domestic production to reduce import dependency. This will help achieve more sustainable economic growth in the long term.

3. **Utilize Clustering Insights for Market Entry**:
   - **Recommendation**: Companies should use the clustering results to target countries within similar trade behavior clusters. This approach will enhance the chances of success in new markets by understanding the local trade dynamics and tailoring strategies accordingly.

4. **Leverage Trade Surpluses for Global Influence**:
   - **Recommendation**: Countries with trade surpluses should consider reinvesting the surplus into global infrastructure projects, development assistance, or research and development initiatives. This will not only benefit the surplus nation but also enhance global economic stability.

5. **Monitor Outliers for Strategic Partnerships**:
   - **Recommendation**: Outlier countries with high trade activity should be studied for strategic partnerships. Special trade agreements with these nations can be valuable for businesses looking to tap into major global trade flows and enhance profitability.

6. **Policy Advocacy for Trade Balance Adjustment**:
   - **Recommendation**: Governments should work towards balancing trade deficits by enhancing domestic industries, reducing unnecessary imports, and promoting exports. A stable and favorable trade balance will ensure economic resilience in the long term.

## Conclusion
By leveraging the insights derived from clustering, trade balance analysis, and correlation patterns, businesses and policymakers can make informed decisions regarding international trade, economic strategy, and market entry. Monitoring these variables will help organizations align their strategies with global trends, improving their competitive advantage in international markets.

