#Functions to load libraries and data
EnsurePackage<-function(x)
{
  x<-as.character(x)
  #
  if (!require(x,character.only=TRUE))
  {
    install.packages(pkgs=x,dependencies = TRUE)
    require(x,character.only=TRUE)
  }
}

load_required_libraries <- function()
{
  EnsurePackage("stringr")
  EnsurePackage("textcat")
  EnsurePackage("stopwords")
  EnsurePackage("text2vec")
  EnsurePackage("xgboost")
  EnsurePackage("caret")
  EnsurePackage("dplyr")
  EnsurePackage("SnowballC")
  EnsurePackage("pROC")
  #EnsurePackage("dbscan")
  EnsurePackage("splitstackshape")
  EnsurePackage("randomForest")
  EnsurePackage("e1071")
}

load_data <- function()
{
  data_path <- "../data/RDS/instagram.rds"
  instagram <- ""
  if(file.exists(data_path))
    instagram <- readRDS(data_path) else
    {
        print("Loading Data from CSV")
        instagram <- read.csv("../data/10k-header.csv",encoding = "UTF-8")
        #Get the image urls for the instagram records
        #instagram <- get_image_url(instagram)
        #Get Class labels for the instagram records
        instagram <- get_Class_quantile(instagram)
        #instagram <- read.csv("../data/10k-header.csv")
        saveRDS(instagram,"../data/RDS/instagram.rds")
    }
  return(instagram)
}

