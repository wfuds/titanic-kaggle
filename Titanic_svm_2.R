# Note: this script is paraphrasing of the SVM script 
# found at http://rstudio-pubs-static.s3.amazonaws.com/16086_88c8db0d8a0148fb8dec4c46961626d5.html

# SVM script for Titanic dataset 

trainData <- read.csv("~/titanic_train.csv", header = TRUE)
testData <- read.csv("~/titanic_test.csv", header = TRUE)

#install.packages("e1071") #Download package for SVM 
library(e1071)

trainData$Survived <- as.factor(trainData$Survived)
trainData$Pclass <- as.factor(trainData$Pclass)

# Remove people with missing age attributes from training and test sets 

key <- complete.cases(trainData$Age)
trainData2 <- trainData[key,]

testData$Age[is.na(testData$Age)] = 29.7

# Use a technique called ten-fold cross-validation to determine our optimal cost value 
# Cost tells us how "soft" we want our margins to be 

trainData2$family = trainData2$SibSp + trainData2$Parch
trainData2$family[trainData2$family > 0] = 1

trainData2$family = as.factor(trainData2$family )

tune.out = tune(svm, Survived ~ Pclass 
                + Sex + Sex * Pclass + family + Age, 
                data = trainData2, kernel = "linear", 
                ranges = list(cost = c(0.001, 0.01, 0.1, 1, 5, 10, 100)))

summary(tune.out)

bestmod = tune.out$best.model
summary(bestmod)

testData$family = testData$SibSp + testData$Parch
testData$family[testData$family > 0] = 1


# Pump out prediction 
yhat.svm.linear = predict(bestmod, testData)

# Creating CSV for Kaggle Submission
survival.svm.linear <- vector()
survival.svm.linear = ifelse(yhat.svm.linear > 0.5, 1, 0)

PassengerId <- testData$PassengerId

kaggle.sub <- cbind(PassengerId, survival.svm.linear)
colnames(kaggle.sub) <- c("PassengerId", "Survived")
write.csv(kaggle.sub, file = "~/titanic_predictions2.csv", 
          row.names = FALSE)
