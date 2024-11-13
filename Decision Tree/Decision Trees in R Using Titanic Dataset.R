# Decision Trees in R
# Install and load necessary packages
install.packages("titanic")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("caret")

library(titanic)
library(rpart)
library(rpart.plot)
library(caret)

getwd() # Function to know the current wd.
setwd("D:\\JALPA\\R-Progamming\\Machine Learning_R") # Function to reset the current wd.

# Loading the Titanic Dataset
# Load dataset
data <- read.csv("D:/JALPA/R-Progamming/Machine Learning_R/DataSet/Titanic.csv")

## OR ##

# Load the Titanic dataset and preprocess it
data("titanic_train")
titanic_data <- titanic_train

## Preprocessing
# Selecting relevant columns and handling missing data
titanic_data <- titanic_data[, c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked")]
titanic_data <- na.omit(titanic_data)  # Removing rows with missing values
titanic_data$Survived <- as.factor(titanic_data$Survived)
titanic_data$Pclass <- as.factor(titanic_data$Pclass)
titanic_data$Sex <- as.factor(titanic_data$Sex)
titanic_data$Embarked <- as.factor(titanic_data$Embarked)

# Split the data into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(titanic_data$Survived, p = 0.7, list = FALSE)
trainData <- titanic_data[trainIndex, ]
testData <- titanic_data[-trainIndex, ]

# Build the Decision Tree model (CART)
tree_model <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = trainData, method = "class")

# Visualize the tree
rpart.plot(tree_model)

# Pruning the Tree to Avoid Overfitting
# We’ll use the complexity parameter (CP) to prune the tree based on cross-validation.
# Displaying the CP table
printcp(tree_model)

# Pruning based on the optimal CP value
optimal_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
pruned_model <- prune(tree_model, cp = optimal_cp)

# Visualizing the pruned tree
rpart.plot(pruned_model)

# Model Validation and Performance Evaluation
# - Finally, let’s evaluate model performance using accuracy, precision, recall, 
#and the confusion matrix.

# Make predictions on the test set
predictions <- predict(pruned_model, newdata = testData, type = "class")

# Confusion matrix to assess performance
conf_matrix <- confusionMatrix(predictions, testData$Survived)

# Print confusion matrix and performance metrics
print(conf_matrix)




