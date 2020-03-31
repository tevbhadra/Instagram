rm(list=ls())
setwd("~/qcri_instagram/src")
source("setup.R")
source("labelling.R")
source("featurize.R")

load_required_libraries()
instagram <- readRDS("../data/RDS/instagram.rds")
#instagram <- splitstackshape::stratified(instagram,"type3_Class",500)
data_matrix <- generate_data_matrix(instagram)

#Classification Model
#XGB - XGBoost
#NB - Naive Bayes
#RF - Random Forest
#SVM - SVM
classifier <- "XGB"
type_Class <- 3
no_folds=5
no_cv=10

Class <- ""
if(type_Class==1)
  Class <- instagram$type1_Class else
    if(type_Class==2)
      Class <- instagram$type2_Class else
        Class <- instagram$type3_Class
Class <- factor(Class)
numberOfClasses = 4
performance <- cross_fold(classifier,data_matrix,Class,no_folds,no_cv,numberOfClasses)
mean_auc <- mean(performance$auc)

mean_precision1 <- mean(performance$precision1)
mean_precision2 <- mean(performance$precision2)
mean_precision3 <- mean(performance$precision3)
mean_precision4 <- mean(performance$precision4)

mean_recall1 <- mean(performance$recall1)
mean_recall2 <- mean(performance$recall2)
mean_recall3 <- mean(performance$recall3)
mean_recall4 <- mean(performance$recall4)

mean_fscore1 <- mean(performance$fscore1)
mean_fscore2 <- mean(performance$fscore2)
mean_fscore3 <- mean(performance$fscore3)
mean_fscore4 <- mean(performance$fscore4)

mean_accuracy1 <- mean(performance$accuracy1)
mean_accuracy2 <- mean(performance$accuracy2)
mean_accuracy3 <- mean(performance$accuracy3)
mean_accuracy4 <- mean(performance$accuracy4)

confusion_matrix_metrics <- data.frame()
mean_precision <- list(mean_precision1,mean_precision2,mean_precision3,mean_precision4)
mean_recall <- list(mean_recall1,mean_recall2,mean_recall3,mean_recall4)
mean_fscore <- list(mean_fscore1,mean_fscore2,mean_fscore3,mean_fscore4)
mean_accuracy <- list(mean_accuracy1,mean_accuracy2,mean_accuracy3,mean_accuracy4)

confusion_matrix_metrics <- do.call("rbind",list(mean_precision,mean_recall,mean_fscore,mean_accuracy))
colnames(confusion_matrix_metrics)<-c("Class_1","Class_2","Class_3","Class_4")
rownames(confusion_matrix_metrics)<-c("Mean Precision","Mean Recall","Mean F-score","Mean Accuracy")

print(paste0("Mean AUC=",mean_auc))
print(confusion_matrix_metrics)

result <- cbind(confusion_matrix_metrics,"Mean AUC" = mean_auc)
write.csv(result,paste0("../data/output/",classifier,"_performance_metrics.csv"))

