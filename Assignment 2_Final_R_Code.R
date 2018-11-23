library(ggplot2) # plotting
library(reshape2) # data wrangling!
library(Matrix)
library(h2o)
library(caret)
library(e1071)
library(dplyr)
library(randomForest)
library(nnet)
library(rpart)
library(parallel)
library(doParallel)

# register multi-core processing
nThreads <- detectCores(logical = TRUE)  # how many threads
cl <- makeCluster(nThreads)  # make a cluster
registerDoParallel(cl)  # register the cluster


# import feature vectors
tfidf_training <- readMM("tfidf_features_training.mtx")
tfidf_testing <- readMM("tfidf_features_testing.mtx")

# import responses and IDs
training_response <- read.csv('training_labels_final.txt',header=FALSE, sep=" ")
names(training_response) <- c('ID','Response')

testing_id <- read.csv('testing_id.csv')  # import the testing ID - they can't be saved in the feature vectors without making them dense which is too large


# utility function to calculate macro F1 score
f1_macro <- function(data,lev=None,model=None){
  
  conf <- confusionMatrix(as.factor(data$obs),as.factor(data$pred))
  f1 <- conf$byClass[,"F1"]
  f1[is.nan(f1)] <- 0
  f1_macro <- mean(f1)
  return(f1_macro)
  
}



set.seed(12345) # save the random seed to make the results reproducble

# set up CV to check our results against 10 folds
ctrl <- trainControl(method = "cv", number = 10, verboseIter=TRUE)

# we run the SVM model with the parameters we optimised using grid search and 5 fold CV
SVM_model <- svm(x=tfidf_training,
                 y=as.factor(training_response[,"Response"]), 
                 gamma=0.00001,cost=1000,
                 kernal='sigmoid',
                 type='C-classification',
                 cross=5)

write.svm(SVM_model, svm.file = "final_model.svm")  # save the model to reduce run time when re-producing


# graph the output of CV
SVM_model.cv


# predict the values
test_model <- SVM_model.predict(tfidf_testing)
test_data <- sapply(test_model,function(x){concat("C",x)})  # add the "c" back to the predictor categories
test_df <- cbind(testing_id,test_data)

# output the predictions to file
write.csv(test_df,"testing_labels_final.txt",sep=" ")

