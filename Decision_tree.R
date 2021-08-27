#Load required packages

library(ggplot2)
library(caret)
library(mlbench)
library(randomForest)
library(MASS)
library(nnet)
library(rattle)
library(MLeval) # for roc calculation
library(pROC)


#Data partition into training and test set using Caret package

data <-read.table("CellLines_52samples_ExprData_tpose2.txt", sep=",", header=TRUE, check.names=F, row.names=1)

#Check dimension of data
dim(data)
head(data[1:10])

set.seed(7)

# Create Training and Test set
training_setIndex <- createDataPartition(data$class, p=0.80, list = FALSE)
# Create Training Data 
training_set <- data[training_setIndex,]

test_set <- data[-training_setIndex,]


#Write into output files
write.table(training_set, file = "train_set.txt", sep="\t", quote=F, row.names = F)

write.table(test_set, file = "test_set.txt", sep="\t", quote=F, row.names = F)

train_control <- trainControl(method="repeatedcv", number=5, repeats=3, classProbs=TRUE, savePredictions = TRUE)


dec_tree <- train(class ~ ., data=training_set, method="rpart", trControl = train_control)
#dec_tree <- train(class ~ ., data=training_set, method="rpart", trControl = train_control, metric = "ROC")


train_results <- dec_tree$results
train_predictions <- dec_tree$pred [1:8]

best_parameters <- dec_tree$bestTune

# Summary of model
summary(dec_tree$finalModel)

#plot decision tree with accuracy at diffferent complexity parameters
plot(dec_tree)

# Print the best tuning parameter cp that maximizes the model accuracy
dec_tree$bestTune

# Plot the final tree model
plot(dec_tree$finalModel, uniform=TRUE, main="Classification Tree")
 text(dec_tree$finalModel, use.n.=TRUE, all=TRUE, cex=.8)
 
# draw colorful decision tree
fancyRpartPlot(dec_tree$finalModel)

test_set1 <- test_set[2:ncol(test_set)]


# Test predictions
test_prediction <- predict(dec_tree, newdata = test_set1)

test_prediction

#Create table
table(test_prediction, test_set$class)

# Compute model accuracy rate on test data
test_accuracy <- round(mean(test_prediction == test_set$class),2)
test_accuracy
#Error rate
error.rate = round(mean(test_prediction != test_set$class),2)
error.rate

#Performance measures on test data

Results <- confusionMatrix(data = as.factor(test_prediction), reference = as.factor(test_set$class))

Results1 <- confusionMatrix(data = as.factor(test_prediction), reference = as.factor(test_set$class), mode = "prec_recall")

Accuracy <- Results$overall

Confusion_mat <- Results$table

Perf_measures <- Results$byClass

library(pROC)

#Roc plot
pred <- predict(dec_tree, newdata=test_set1, type="prob")
roc <- evalm(data.frame(pred, test_set$Class))


## ROC plot for train
pred_tr <- predict(dec_tree, newdata=training_set[2:ncol(training_set)], type="prob")

roc_tr <- evalm(data.frame(pred_tr, training_set$Class))

pred
