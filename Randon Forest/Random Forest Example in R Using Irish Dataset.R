# Random Forest Example in R
# Load Necessary Libraries
install.packages("randomForest")
install.packages("caret")  # for tuning parameters and validation
library(randomForest)
library(caret)

# Load and Explore the Dataset
data(iris)
summary(iris)

# Split the Data into Training and Testing Sets
# Setting a seed for reproducibility
set.seed(123)

# Create a 70-30 train-test split
trainIndex <- createDataPartition(iris$Species, p = 0.7, list = FALSE)
trainData <- iris[trainIndex, ]
testData <- iris[-trainIndex, ]

# Train a Random Forest Model
# Basic Random Forest model
rf_model <- randomForest(Species ~ ., data = trainData, ntree = 100, mtry = 2)
print(rf_model)

# Tune Hyperparameters of Random Forest
# Tuning 'mtry' parameter using caret
tuneGrid <- expand.grid(.mtry = c(1:4))  # Possible values for mtry

# TrainControl for cross-validation
control <- trainControl(method = "cv", number = 5)

# Train with cross-validation
rf_tuned <- train(Species ~ ., data = trainData, method = "rf",
                  tuneGrid = tuneGrid, trControl = control, ntree = 100)
print(rf_tuned)

# Evaluate Model Performance on Test Data
# Predict on test data using tuned model
predictions <- predict(rf_tuned, testData)
confusionMatrix(predictions, testData$Species)

# Assess Feature Importance
# Variable importance plot
importance(rf_model)
varImpPlot(rf_model)

# Plotting the Model
# Error rate vs number of trees
plot(rf_model)






