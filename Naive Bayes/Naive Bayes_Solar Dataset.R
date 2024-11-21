#installing required packages
install.packages('mlbench')
install.packages("caret", dependencies = c("Depends", "Suggests"))
install.packages('e1071')

# importing the required libraries
library(mlbench)
library(caret)
library(e1071)

# Loading the dataset
data("Sonar")
# printing the first few rows of the dataset
head(Sonar)

# Splitting the dataset
set.seed(123)  # for reproducibility
splitIndex <- createDataPartition(Sonar$Class, p = 0.7, 
                                  list = FALSE, 
                                  times = 1)
train_data <- Sonar[splitIndex, ]
test_data <- Sonar[-splitIndex, ]

# training the model
model <- naiveBayes(Class ~ ., data = train_data)

# making predictions
predictions <- predict(model, newdata = test_data)

# evaluating the model
confusion_matrix <- table(predictions, test_data$Class)
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
print(confusion_matrix)
cat("Accuracy:", round(accuracy * 100, 2), "%\n")


# Extracting particular test dataset entry
entry <- test_data[5, ]

# Make a prediction using the Naive Bayes model
prediction <- predict(model, newdata = entry)

# Print the actual class and the predicted class
cat("Actual Class:", entry$Class, "\n")
cat("Predicted Class:", prediction, "\n")
