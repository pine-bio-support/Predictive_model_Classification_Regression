#Load libraries
library(randomForest)
library(BBmisc)
library(pROC)

set.seed(10)

options(stringsAsFactors = FALSE)
numtrees <- as.numeric(500)

#Prepare training table
trainset <- read.table("https://code.omicslogic.com/assets/datasets/cell_lines/CellLines_52samples_ExprData_T1.txt", sep='\t', header=FALSE, stringsAsFactors=FALSE)

#transpose
id_n = 1
trainset = data.frame(t(trainset))

#define column names (data has been transposed, so these are genes)
cnam <- t(trainset)[,1]
trainset <- trainset[-1,]
colnames(trainset) <- cnam

#define row names (data has been transposed, so these are names of samples)
rnam <- trainset[,id_n]
trainset <- trainset[,-id_n]
rownames(trainset) <- rnam

#define numeric data
trainset[,1]<- as.factor(trainset[,1])
len= dim (trainset)[2]
for (j in (2:len)){
  trainset [,j]<- as.numeric(trainset[,j])
}


#Prepare testing dataset - transpose if required, etc.
testset <- read.table("https://code.omicslogic.com/assets/datasets/cell_lines/CellLines_52samples_ExprData_T1.txt", sep="\t", header=FALSE, stringsAsFactors=FALSE)
testset <- data.frame(t(testset))

#define column names
cnam <- t(testset)[,1]
testset <- testset[-1,]
colnames(testset) <- cnam

#define row names
rnam <- testset[,1]
testset <- testset[,-1]
rownames(testset) <- rnam

len <- dim (testset)[2]
for (j in (1:len)){
  testset [,j]<- as.numeric(testset[,j])
}

#Now we can build the model
#Perform main computations - train classifiers
trainset [,1] <- factor(trainset[,1])
rforest <- randomForest(class ~ ., data=trainset, ntree=numtrees, importance=TRUE)

#Draw plots
plot (rforest, main='Random forest plot', lty=1, cex=1.5)

#Feature importance
#Store features in a vaiable as features
features <- varImpPlot(rforest, main='Feature Importance', pch=1, cex=0.6)




#Write features in a txt file format
write.table(features,'features_importance.txt', row.names=TRUE,col.names=NA, sep='\t', quote=FALSE)

# Sort features based on meandecrease accuracy value
sorted_features <- as.data.frame(features[order(features[,"MeanDecreaseAccuracy"], decreasing = TRUE),])

#select only top features (meanDecreaseaccuracy >1.5)
top_features <- sorted_features[sorted_features$MeanDecreaseAccuracy>1.5, ]




#Apply classifier to the test set
prediction <- predict(rforest, testset, type='prob')
prediction <- data.frame(prediction, check.names = FALSE)
rownames(prediction) <- rownames(testset)

#Print out prediction values
a <- print(prediction)
write.table(a, file = "prediction_class.txt", sep="\t", quote=F, row.names = T)

#AUC and ROC curves
roc1 <- multiclass.roc(trainset$class,rforest$votes[,1], percent=TRUE, grid=TRUE, print.auc=TRUE, col='red', direction='>', plot=TRUE, main="ROC curve")
print(roc1)

#print ROC plots
for(i in c(1,2,3,4)){
  plot(roc1$rocs[[i]], colour = "green", main = roc1$levels[[i]], percent=TRUE, grid=TRUE, print.auc=TRUE, col='red')
}
