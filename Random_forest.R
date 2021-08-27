Load required packages

library(ggplot2)
library(lattice)
library(caret)
library(C50)
library(kernlab)
library(mlbench)
library(randomForest)
library(caretEnsemble)
library(MASS)
library(klaR)
library(nnet)
library(rattle)


#Data partition into training and test set using Caret package

data <-read.table("CellLines_52samples_ExprData_tpose2.txt", sep=",", header=TRUE, check.names=F, row.names=1)

#Check dimension of data
dim(data)
head(data[1:10])

set.seed(17)

# Create Training and Test set
training_setIndex <- createDataPartition(data$class, p=0.80, list = FALSE)
# Create Training Data 
training_set <- data[training_setIndex,]

test_set <- data[-training_setIndex,]


#Write into output files
write.table(training_set, file = "train_set.txt", sep="\t", quote=F, row.names = F)

write.table(test_set, file = "test_set.txt", sep="\t", quote=F, row.names = F)

tr_control <- trainControl(method='repeatedcv',  number=5,  repeats=3, search='grid')

tunegrid <- expand.grid(.mtry = (1:15)) 


rf_gridsearch <- train(class ~ ., data = training_set, method = 'rf', metric = 'Accuracy', tuneGrid = tunegrid)

rf_model = rf_gridsearch 

train_results <- rf_model$results
train_predictions <- rf_model$pred [1:8]

best_parameters <- rf_model$bestTune

# Summary of model
summary(dec_tree$finalModel)

#plot decision tree with accuracy at diffferent complexity parameters
plot(rf_model)

# Print the best tuning parameter cp that maximizes the model accuracy
rf_model$bestTune

# Plot the final tree model
plot(rf_model$finalModel, uniform=TRUE, main="Random Forest")

#Prepare test data
test_set1 <- test_set[2:ncol(test_set)]


# Test predictions
test_prediction <- predict(rf_model, newdata = test_set1)

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
Results

Results1 <- confusionMatrix(data = as.factor(test_prediction), reference = as.factor(test_set$class), mode = "prec_recall")
Results1
Accuracy <- Results$overall
Accuracy
Confusion_mat <- Results$table
Confusion_mat
Perf_measures <- Results$byClass
Perf_measures

library(pROC)


tr_pred <- predict(rf_model, newdata=training_set[2:ncol(training_set)])
tr_pred 

roc_tr <-  multiclass.roc(training_set$class,as.numeric(tr_pred), percent=TRUE, plot=TRUE, print.auc=TRUE)
roc_tr

roc_te <-  multiclass.roc(test_set$class,as.numeric(test_prediction), percent=TRUE, plot=TRUE, print.auc=TRUE)
roc_te









print(roc1)

#Roc plot
pred <- predict(rf_model, newdata=test_set1, type="prob")
roc <- evalm(data.frame(as.factor(test_set$Class), as.factor(test_prediction)))


## ROC plot for train
pred_tr <- predict(rf_model, newdata=training_set[2:ncol(training_set)])

roc_tr <- evalm(data.frame( training_set$Class, pred_tr))

