#Functions to compute engagement rate and identify the classes of the dataset
get_Class_quantile <- function(instagram)
{
  instagram$er_likes <- compute_engagement_likes(instagram$likes,
                                                 instagram$user_followed_by)
  er_quantile <- quantile(instagram$er_likes)
  instagram$type1_Class[instagram$er_likes >= er_quantile[1] 
                           & instagram$er_likes < er_quantile[2]] <- 1
  instagram$type1_Class[instagram$er_likes >= er_quantile[2] 
                           & instagram$er_likes < er_quantile[3]] <- 2
  instagram$type1_Class[instagram$er_likes >= er_quantile[3] 
                           & instagram$er_likes < er_quantile[4]] <- 3
  instagram$type1_Class[instagram$er_likes >= er_quantile[4] 
                           & instagram$er_likes <= er_quantile[5]] <- 4
  
  #**********************************************************************************
  instagram$er_comments <- compute_engagement_comments(instagram$comments,
                                                       instagram$user_followed_by)
  er_quantile <- quantile(instagram$er_comments)
  instagram$type2_Class[instagram$er_comments >= er_quantile[1] 
                           & instagram$er_comments < er_quantile[2]] <- 1
  instagram$type2_Class[instagram$er_comments >= er_quantile[2] 
                           & instagram$er_comments < er_quantile[3]] <- 2
  instagram$type2_Class[instagram$er_comments >= er_quantile[3] 
                           & instagram$er_comments < er_quantile[4]] <- 3
  instagram$type2_Class[instagram$er_comments >= er_quantile[4] 
                           & instagram$er_comments <= er_quantile[5]] <- 4
  #***********************************************************************************
  instagram$er_likes_comments <- compute_engagement_likes_comments(instagram$comments+instagram$likes,
                                                       instagram$user_followed_by)
  er_quantile <- quantile(instagram$er_likes_comments)
  instagram$type3_Class[instagram$er_likes_comments >= er_quantile[1] 
                           & instagram$er_likes_comments < er_quantile[2]] <- 1
  instagram$type3_Class[instagram$er_likes_comments >= er_quantile[2] 
                           & instagram$er_likes_comments < er_quantile[3]] <- 2
  instagram$type3_Class[instagram$er_likes_comments >= er_quantile[3] 
                           & instagram$er_likes_comments < er_quantile[4]] <- 3
  instagram$type3_Class[instagram$er_likes_comments >= er_quantile[4] 
                           & instagram$er_likes_comments <= er_quantile[5]] <- 4
  return(instagram)
}
compute_engagement_likes <- function(likes,followed_by)
{
  engagement_rate <- likes / log(followed_by)
  return(engagement_rate)
}
compute_engagement_comments <- function(comments,followed_by)
{
  engagement_rate <- comments / tan(followed_by)
  return(engagement_rate)
}
compute_engagement_likes_comments <- function(likes_comments,followed_by)
{
  engagement_rate <- likes_comments / log(followed_by)
  return(engagement_rate)
}
get_Class_threshold <- function(instagram)
{
  print("Labelling based on Thresholds")
  instagram$engagement_score <- compute_engagement_likes_comments(instagram$comments+instagram$likes,
                                                                   instagram$user_followed_by)
  instagram$Class[instagram$engagement_score<=400] <- "1"
  instagram$Class[instagram$engagement_score>400 & instagram$engagement_score<=800] <- "2"
  instagram$Class[instagram$engagement_score>800 & instagram$engagement_score<=1200] <- "3"
  instagram$Class[instagram$engagement_score>1200] <- "4"
  return(instagram)
}


