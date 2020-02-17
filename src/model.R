# Function to build the classification model
XGB_classifier <- function(train_matrix,train_Class,numberOfClasses)
{
  #Convert the Class factor to an integer class starting at 0
  #Requirement in xgboost
  xgb_Class = as.integer(train_Class)-1
  #Prepare the data matrix for xgboost
  train = data.matrix(train_matrix)
  class(train) <- "numeric"
  xgb_train = xgb.DMatrix(data=train,label=xgb_Class)
  #Train the model
  numberOfClasses <- length(unique(Class))
  xgb_params <- list("objective" = "multi:softprob",
                     "eval_metric" = "mlogloss",
                     "num_class" = numberOfClasses)
  print("   Training XGBoost Model")
  model <- xgb.train(params = xgb_params, data = xgb_train, nround=50)
  return(model)
}
RF_classifier <- function(train_matrix,train_Class,numberOfClasses)
{
  print("    Training Random Forest Model")
  model <- randomForest(x = train_matrix,y = as.factor(train_Class))
  return(model)
}
NB_classifier <- function(train_matrix,train_Class,numberOfClasses)
{
  print("    Training Naive bayes model")
  model <- naiveBayes(x = train_matrix,y = as.factor(train_Class))
  return(model)
}
SVM_classifier <- function(train_matrix,train_Class,numberOfClasses)
{
  print("    Training SVM model")
  train_matrix <- data.matrix(train_matrix)
  class(train_matrix) <- "numeric"
  model <- svm(x = train_matrix,y = as.factor(train_Class),probability = TRUE)
  return(model)
}
predict_on_datamatrix <- function(model,classifier,test_matrix)
{
  print("   Prediction on Test data")
  if(classifier == "XGB")
  {
    #Prepare the test data for XGB prediction
    test = data.matrix(test_matrix)
    class(test) <- "numeric"
    xgb_test = xgb.DMatrix(data=test)
    #Prediction
    pred <- predict(model, newdata = xgb_test)
  }else if(classifier == "RF")
  {
    test = data.matrix(test_matrix)
    pred <- predict(model, newdata = test,type = "prob")
  }else if(classifier == "NB")
  {
    test = data.matrix(test_matrix)
    pred <- predict(model, newdata = test,type = "raw") 
  }else if(classifier == "SVM")
  {
    test = data.matrix(test_matrix)
    class(test) <- "numeric"
    pred <- predict(model, newdata = test,probability = TRUE)
  }
  return(pred)
}

evaluate_performance <- function(model,classifier,test_matrix,test_Class,numberOfClasses)
{
  pred <- predict_on_datamatrix(model,classifier,test_matrix)
  if(classifier == "SVM")
  {
    pred <- attr(pred, "probabilities")
    pred <- pred[,order(colnames(pred))]
  }
  print("   Computing Performace Metrics")
  test_prediction <- ""
  if(classifier == "XGB")
  {
    test_prediction <- matrix(pred, nrow = numberOfClasses,
                              ncol=length(pred)/numberOfClasses) %>%
      t() %>%
      data.frame() %>%
      mutate(actual = test_Class,
             predicted = max.col(., "last"))
  }else
  {
    test_prediction <- list()
    test_prediction$actual <- test_Class
    test_prediction$predicted <- max.col(pred)
  }
  
  confusionMatrix <- caret::confusionMatrix(factor(test_prediction$predicted),
                                            factor(test_prediction$actual),
                                            mode = "everything")
  c <- numberOfClasses * 4
  Precision_1 <- confusionMatrix$byClass[c+1]
  Precision_2 <- confusionMatrix$byClass[c+2]
  Precision_3 <- confusionMatrix$byClass[c+3]
  Precision_4 <- confusionMatrix$byClass[c+4]
  
  Recall_1 <- confusionMatrix$byClass[c+5]
  Recall_2 <- confusionMatrix$byClass[c+6]
  Recall_3 <- confusionMatrix$byClass[c+7]
  Recall_4 <- confusionMatrix$byClass[c+8]
  
  Fscore_1 <- confusionMatrix$byClass[c+9]
  Fscore_2 <- confusionMatrix$byClass[c+10]
  Fscore_3 <- confusionMatrix$byClass[c+11]
  Fscore_4 <- confusionMatrix$byClass[c+12]
  
  c <- numberOfClasses * 10
  Accuracy_1 <- confusionMatrix$byClass[c+1]
  Accuracy_2 <- confusionMatrix$byClass[c+2]
  Accuracy_3 <- confusionMatrix$byClass[c+3]
  Accuracy_4 <- confusionMatrix$byClass[c+4]
  
  column_names <- sort(unique(test_Class))
  #Compute AUC for multi-class
  auc <- ""
  if(classifier == "XGB")
  {
    pred_matrix <- matrix(pred, nrow = numberOfClasses,
                          ncol=length(pred)/numberOfClasses) %>%
      t() %>%
      data.frame()
    colnames(pred_matrix) <- column_names
    auc <- pROC::multiclass.roc(test_Class,pred_matrix)$auc
  }else
  {
    auc <- pROC::multiclass.roc(test_Class,pred)$auc
  }
  
  
  perf <- list("auc" = auc,
               "precision1"= Precision_1,"recall1"=Recall_1,"fscore1"=Fscore_1,"accuracy1"=Accuracy_1,
               "precision2"=Precision_2,"recall2"=Recall_2,"fscore2"=Fscore_2,"accuracy2"=Accuracy_2,
               "precision3"=Precision_3,"recall3"=Recall_3,"fscore3"=Fscore_3,"accuracy3"=Accuracy_3,
               "precision4"=Precision_4,"recall4"=Recall_4,"fscore4"=Fscore_4,"accuracy4"=Accuracy_4)
  return(perf)
}
cross_fold <- function(classifier,data_matrix,Class,no_folds,no_cv,numberOfClasses)
{
  Performance <- data.frame()
  
  #Bind data matrix and Class before splitting into folds
  data_matrix <- cbind(data_matrix,Class)
  for(i in 1:no_cv)
  {
    folds <- caret::createFolds(Class,k=no_folds)
    for(j in 1:length(folds))
    {
      print(paste0("Cross Fold Round: ",i," Fold: ",j))
      test_fold <- folds[[j]]
      remaining_folds <- folds[-j]
      train_fold <- list()
      for(k in 1:length(remaining_folds))
      {
        train_fold <- append(train_fold,remaining_folds[[k]])
      }
      train_fold <- unlist(train_fold)
      train_data <- data_matrix[train_fold,]
      test_data <- data_matrix[test_fold,]
      
      train_Class <- train_data[,ncol(train_data)]
      train_matrix <- train_data[,-ncol(train_data)]
      test_Class <- test_data[,ncol(test_data)]
      test_matrix <- test_data[,-ncol(test_data)]
      
      model <- tryCatch(do.call(paste0(classifier,"_classifier"),
                                list(train_matrix,train_Class,numberOfClasses)),
                          error=function(e)
                          {
                            print("Error Handling")
                            print(e)     
                            return(model=NULL)
                          })
      if(!is.null(model))
          perf <- evaluate_performance(model,classifier,test_matrix,test_Class,numberOfClasses)
      Performance <- rbind(Performance,perf)
    }
  }
  return(Performance)
}
