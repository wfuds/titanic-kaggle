# Using the Random Forest technique to predict on the Titanic dataset 
# Note: this script based roughly on tutorial found at 
# http://trevorstephens.com/post/73770963794/titanic-getting-started-with-r-part-5-random
library(randomForest)

# Read in data ----
trainData <- read.csv("~/titanic_train.csv", header = TRUE)
testData <- read.csv("~/titanic_test.csv", header = TRUE)

# Let's clean and transform the data ----

# Create family variable 
trainData$family = trainData$SibSp + trainData$Parch
trainData$family = ifelse(trainData$family > 0, 1, 0) # Hopefully not too much variability lost here 

testData$family = testData$SibSp + testData$Parch
testData$family = ifelse(testData$family > 0, 1, 0)

# Fill in missing age data using Random Forests 

agetree = randomForest(Age ~ Survived + Pclass + Sex + Fare + family, data = trainData[!is.na(trainData$Age),])
trainData$Age[is.na(trainData$Age)] <- predict(agetree, trainData[is.na(trainData$Age),])

# For the test set ... 
agetree2 = randomForest(Age ~ Pclass + Sex + Fare + family, data = trainData)
testData$Age[is.na(testData$Age)] <- predict(agetree2, testData[is.na(testData$Age),])

# Patch up missing Embarked data 
trainData$Embarked[trainData$Embarked == ""] = "S"
trainData <- droplevels(trainData)

# Patch up missing Fare in test set 
testData$Fare[is.na(testData$Fare)] = 35.627

# Create Random Forests model ---- 

set.seed(1)
fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + Fare + family, data = trainData, 
          importance = TRUE, ntree = 2000) # Because we have a smaller dataset, running a huge number of trees not an issue 

varImpPlot(fit) # Check which variables actually mattered. MDA tells us how much worse the model performs w/o given variable, 
# and MDG has a similar interpretation. So, embarked not very important. 

# Create predction for Kaggle submission ----

results <- predict(fit, testData)

submit <- data.frame(PassengerId = testData$PassengerId, Survived = results)
write.csv(submit, file = "~/titanic_predictionsRF.csv", row.names = FALSE)
