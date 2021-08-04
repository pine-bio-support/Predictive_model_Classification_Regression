
install.packages("caret")
# load caret
library("caret")
install.packages('Rcpp')
library(Rcpp)


#Data Split
#Prepare training and test data

data <-read.table("CellLines_52samples_ExprData_tpose.txt", sep=",", header=TRUE, check.names=F)

head(data[1:10])
dim(data)

#Split training and test data into 80:20 ratio 

dt = sort(sample(nrow(data), nrow(data)*.8))

train_set<-data[dt,]

dim(train_set)

test_set<-data[-dt,]

dim(test_set)


write.table(train_set, file = "train_set.txt", sep="\t", quote=F, row.names = F)

write.table(test_set, file = "test_set.txt", sep="\t", quote=F, row.names = F)
#### Training Test split using caret package #######

data <-read.table("full_data.txt", sep="\t", header=TRUE, row.names=1)

# use 80% of the original training data for training
set.seed(12345)
train_index <- createDataPartition(data$id, p=0.80, list=FALSE)
#train_set <- train[train_index,]

# use the remaining 20% of the original training data for validation
#test_set <- train[-train_index,]



##Classification - RF

train_set1 <- as.matrix(train_set)

trcontrol = trainControl(method='cv', number=10, savePredictions = T, classProbs = TRUE,summaryFunction = twoClassSummary,returnResamp="all") 

trcontrol1 = trainControl(method="cv", number=10, repeats=3, classProbs= TRUE, summaryFunction = multiClassSummary)

classes <- as.factor(train_set$class)

cl <- make.names(class, unique = FALSE, allow_ = TRUE)

model = train(cl ~ . , data=train_set, method = "rf", trControl = trcontrol1, metric="ROC") 

model$resample


