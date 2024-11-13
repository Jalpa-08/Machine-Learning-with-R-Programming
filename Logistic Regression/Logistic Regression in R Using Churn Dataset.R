# Logistic Regression in R
# 1. Dataset Preparation

# Load necessary packages
install.packages("dplyr")
install.packages("InformationValue")
install.packages("car")
library(dplyr)
library(InformationValue)
library(car)

# Load the churn dataset
df <- read.csv("D:/JALPA/R-Progamming/Machine Learning_R/DataSet/sample_churn_data.csv")  # Update with your file path

# Convert the Churn column to a binary variable (1 for "Yes", 0 for "No")
df$Churn <- ifelse(df$Churn == "Yes", 1, 0)
# View the first few rows
head(df)

# 2. Likelihood Profiling
# In logistic regression, likelihood profiling involves estimating parameters 
#using maximum likelihood estimation (MLE).

# Fitting a basic logistic regression model
model <- glm(Churn ~ ., data = df, family = "binomial")
summary(model)

# 3. Assumptions Check
# Logistic regression assumes a linear relationship between predictors and the 
#log-odds of the outcome. To check for multicollinearity, calculate the Variance Inflation Factor (VIF):

# Check for multicollinearity using VIF
# Compute correlation matrix for numeric variables
numeric_vars <- df %>% select_if(is.numeric)
cor_matrix <- cor(numeric_vars)

# Display correlation matrix to identify correlations above a threshold (e.g., 0.8)
print(cor_matrix)

# Refit model after removing correlated variables (adjust list as needed)
model <- glm(Churn ~ Gender + SeniorCitizen + Tenure + MonthlyCharges, data = df, family = "binomial")

# Check VIF again
vif_values <- vif(model)
print(vif_values)

# Stepwise selection for variable reduction
library(MASS)
model <- stepAIC(model, direction = "both")
summary(model)

# 4. Variable Selection with WoE and IV
# Using Weight of Evidence (WoE) and Information Value (IV) can help with 
#feature selection by identifying the predictive power of categorical variables.

# Calculate WoE and IV
install.packages("Information")
library(Information)

iv <- create_infotables(data = df, y = "Churn", bins = 10, parallel = TRUE)
print(iv$Summary)  # Summary of IV values for each variable

# 5. Model Validation
# To validate the model, split the data into training and test sets, then train 
#the model on the training set.

# Split data into training and test sets (70-30 split)
set.seed(123)
train_indices <- sample(seq_len(nrow(df)), size = 0.7 * nrow(df))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

# Train the logistic regression model on the training set
train_model <- glm(Churn ~ ., data = train_data, family = "binomial")

# 6. Model Performance
# Evaluate model performance on the test set using accuracy, sensitivity, 
#specificity, and AUC.

# Predict probabilities on the test set
predictions <- predict(train_model, test_data, type = "response")
test_data$predicted_class <- ifelse(predictions > 0.5, 1, 0)

# Confusion matrix
conf_matrix <- table(test_data$Churn, test_data$predicted_class)
print(conf_matrix)

# Calculate performance metrics
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
sensitivity <- conf_matrix[2, 2] / sum(conf_matrix[2, ])
specificity <- conf_matrix[1, 1] / sum(conf_matrix[1, ])

# Calculate AUC
auc <- InformationValue::AUROC(predictions, test_data$Churn)
cat("Accuracy:", accuracy, "\nSensitivity:", sensitivity, "\nSpecificity:", specificity, "\nAUC:", auc)

# 7. Prediction on New Data
# You can make predictions on new data with the model you’ve trained.

# Predict on new data (example shown with test set here)
new_predictions <- predict(train_model, test_data, type = "response")
test_data$predicted_class <- ifelse(new_predictions > 0.5, 1, 0)

