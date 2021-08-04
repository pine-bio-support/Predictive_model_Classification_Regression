

library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
library(tidyverse)
library(caret)
library(glmnet)
library(Hmisc)
library(e1071)
library(hydroGOF)
require(randomForest)
require(MASS)
library(caTools)


#set seed
set.seed(42)
#Load data
data= read.csv('Gestational_age_features.txt', header = T, sep ='\t', row.names=1)



#Train/test split
dt = sort(sample(nrow(data), nrow(data)*.8))
train_1<- data[dt,]
test_1<- data[-dt,]

#Create regression formula
f <- as.formula(paste(names(train_1)[1], "~", paste(names(train_1)[2:69], collapse=" + ")))

#k-fold cross validation
train.control = trainControl(method = "cv", number = 10)

#Training model
model<- train(f, data = train_1, method = "lm", trControl = train.control)

model

#Save model 
save(mod, file = "Linear_regression_train.rda")	

#Prepare test input
test1  <- test_1[2:69]

test1

# Compute test prediction
test_pred<-predict(model, newdata = test1)

#Create test prediction resulst object
test_res <- cbind(row.names(test_1), test_1$GA, test_pred )

#Provide the column names 
colnames(test_res) <- c("ID", "Actual_GA", "Predicted_GA")

#write into a file
write.table(test_res,file = "test_res_GA.txt", sep ="\t", row.names=F, quote = F)


#Evaluate the performance of model 

# Compute Correlation
cor_lm = cor(test_1$GA,test_pred)

#print correlation
cor_lm


#Calculate NRMSE
rmse_lm = rmse(test_1$GA,test_pred)

#Print NRMSE
rmse_lm

#Calculate MAE
mae_lm = MAE(test_1$GA,test_pred)

#print MAE
mae_lm

#calculate r2
r2_lm = R2(test_1$GA,test_pred)

#print r2
r2_lm

