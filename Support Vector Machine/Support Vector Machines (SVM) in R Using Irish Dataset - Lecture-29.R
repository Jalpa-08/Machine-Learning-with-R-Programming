# Support Vector Machines (SVM) in R
# 1: Install and Load Necessary Libraries
install.packages("e1071")
install.packages("caret")
# Install Rtools Link Below:
https://cran.rstudio.com/bin/windows/Rtools/rtools44/rtools.html

library(e1071)
library(caret)

# 2: Load and Prepare the Dataset
# Load the dataset
data("iris")

# View the first few rows of the dataset
head(iris)

# Check for missing values
sum(is.na(iris))

# 3: Train-Test Split
# Set seed for reproducibility
set.seed(123)

# Split data into training (70%) and testing (30%) sets
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
train_data <- iris[trainIndex, ]
test_data <- iris[-trainIndex, ]

# 4: Train the SVM Model
# Hyperparameters
# - Kernel: Controls how the data is mapped to higher dimensions. Common choices 
#include "linear", "polynomial", "radial" (Gaussian), and "sigmoid".
# - Cost (C): Controls the trade-off between achieving a low training error and 
#a low testing error (regularization).
# - Gamma: Affects the shape of the decision boundary in non-linear kernels 
#(like "radial").

# Build the Model
# Train SVM with radial kernel
svm_model <- svm(Species ~ ., data = train_data, kernel = "radial", cost = 1, gamma = 0.1)

# View model summary
summary(svm_model)

# 5: Model Interpretation and Visualizations
# Hyperplane
# - The SVM classifier aims to find the optimal hyperplane that separates classes. 
#In higher dimensions, this hyperplane may not be a simple line but rather a 
#decision boundary. Visualizing the hyperplane and support vectors can be 
#challenging in high-dimensional data but can be illustrated in a 2D plane if 
#limited to two features.

# Fit a simplified SVM model with two features
svm_model_simple <- svm(Species ~ Sepal.Length + Sepal.Width, data = train_data, kernel = "linear", cost = 1)

# Plot the decision boundary and support vectors
plot(svm_model_simple, train_data, Sepal.Length ~ Sepal.Width)

# Kernels
# - The kernel function is critical for handling non-linearly separable data. 
#In the code above, we used a "radial" kernel to account for more complex boundaries.

# 6: Model Validation
# Predict on Test Data
# Make predictions
predictions <- predict(svm_model, test_data)

# Display the confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$Species)
print(conf_matrix)

# Model Performance Metrics
# - The confusion matrix shows accuracy, sensitivity, specificity, and other 
#metrics to evaluate how well the model performed.

# 7: Hyperparameter Tuning
# Set up training control
train_control <- trainControl(method = "cv", number = 10)

# Set up a grid of values for `C` and `sigma`
tune_grid <- expand.grid(.C = c(0.1, 1, 10), .sigma = c(0.01, 0.1, 1))

# Train SVM model with grid search
svm_tuned <- train(Species ~ ., data = train_data, method = "svmRadial",
                   trControl = train_control, tuneGrid = tune_grid)

# Print the results of the grid search
print(svm_tuned)

# 8: Retrain with Optimal Hyperparameters
# Use the best values for cost and gamma from tuning
best_cost <- svm_tuned$bestTune$C  # C is the cost parameter
best_sigma <- svm_tuned$bestTune$sigma  # sigma corresponds to gamma in radial kernel

# Retrain the SVM model
svm_optimized <- svm(Species ~ ., data = train_data, kernel = "radial", cost = best_cost, gamma = best_sigma)

# View the optimized model summary
summary(svm_optimized)

# 9: Final Evaluation
# Predict and evaluate on test data
final_predictions <- predict(svm_optimized, test_data)
final_conf_matrix <- confusionMatrix(final_predictions, test_data$Species)
print(final_conf_matrix)








