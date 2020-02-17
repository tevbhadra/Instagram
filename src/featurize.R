#Functions to build the features from the data
generate_data_matrix <- function(instagram)
{
  datamatrix_path <- "../data/RDS/datamatrix.rds"
  data_matrix <- ""
  if(file.exists(datamatrix_path))
    data_matrix <- readRDS(datamatrix_path) else
    {
      caption_text <- get_only_text(instagram$caption)
      print("Computing number of characters in the caption")
      no_char <- nchar(caption_text)
      print("Computing number of words in the caption")
      no_word <- stringr::str_count(caption_text,'\\w+')
      print("Identifying the language of the caption")
      language <- identify_language(caption_text)
      print("Generating TFIDF matrix for the caption")
      tfidf_matrix <- generate_tfidf_matrix(caption_text,instagram$id)
      print("Generating Hashtag matrix")
      hashtag_matrix <- featurize_hashtag(instagram$hashtags,instagram$id)
      #image <- featurize_image(instagram$url)
      image_matrix <- read.csv("../data/image_matrix.csv")
      print("Binding all the features together to generate a data matrix")
      data_matrix <- cbind(
                           no_char,
                           no_word,
                           language,
                           as.matrix(tfidf_matrix),
                           as.matrix(hashtag_matrix),
                           as.matrix(image_matrix)
      )
      print("Saving the Data Matrix")
      saveRDS(data_matrix,datamatrix_path)
    }
  return(data_matrix)
}
generate_tfidf_matrix <- function(text_data,id)
{
 text_data <- get_clean_text_data(text_data)
 prep_fun = tolower
 tok_fun = word_tokenizer
 
 stem_tokenizer =function(x) {
   tokens = word_tokenizer(x)
   lapply(tokens, SnowballC::wordStem, language="en")
 }
 
 it = itoken(text_data, 
             preprocessor = prep_fun, 
             tokenizer = stem_tokenizer, 
             ids = id, 
             progressbar = FALSE)
 
 vocab = create_vocabulary(it,stopwords =  c(stopwords::stopwords(language = "english"),
                                             stopwords::stopwords(source = "smart")))
 vocab = prune_vocabulary(vocab, term_count_min = 20,doc_count_min = 10,vocab_term_max = 10000)
 
 #Construct a vector space 
 vectorizer = vocab_vectorizer(vocab)
 #Construct Document Term Matrix from the vectorizer
 dtm = create_dtm(it, vectorizer)
 tfidf = TfIdf$new()
 tfidf_matrix = tfidf$fit_transform(dtm)
 return(tfidf_matrix)
}

get_clean_text_data <- function(text_data)
{
  print("Preprocessing of the caption text")
  text_data = gsub("[[:punct:][:blank:]]+", " ", text_data)
  text_data = gsub("[^\x01-\x7F]", "", text_data)
  text_data = gsub('[0-9]+',"",text_data)
  text_data = tolower(text_data)
  text_data = gsub('\\b\\w{1,2}\\s','',text_data)
  #text_data <- iconv(text_data, 'utf-8', 'ascii', sub='')
  stopWords <- stopwords("english")
  '%nin%' <- Negate('%in%')
  text_data <- lapply(text_data, function(x) {
    t <- unlist(strsplit(x, " "))
    t[t %nin% stopWords]
  })
  return(text_data)
}
get_only_text <- function(caption)
{
  print("Getting only text from the captions. Getting rid of hashtags and symbols")
  caption <- iconv(caption, 'utf-8', 'ascii', sub='')
  caption <- gsub("#([a-zA-Z0-9]|[_])*","",caption)
  caption <- stringr::str_replace_all(caption, "[^[:alnum:]]", " ")
  caption <- trimws(caption,"r")
  return(caption)
}
identify_language <- function(caption)
{
  #0 - no text
  #1 - finnish
  #2 - english
  language <- textcat::textcat(caption)
  for(i in 1:length(language))
    if(is.na(language[i]))
      language[i] <- 0 else
        if(language[i]=="finnish")
          language[i] <- 1 else
            language[i] <- 2
  
  return(language)
}
featurize_hashtag <- function(hashtag,id)
{
  hashtag = gsub(","," ", hashtag)
  hashtag = gsub("[[:punct:][:blank:]]+", " ", hashtag)
  hashtag = gsub("[^\x01-\x7F]", "", hashtag)
  prep_fun = tolower
  tok_fun = word_tokenizer
  
  tokenizer =function(x) {
    tokens = word_tokenizer(x)
    #lapply(tokens, SnowballC::wordStem, language="en")
  }
  
  it = itoken(hashtag, 
              #preprocessor = prep_fun, 
              tokenizer = tokenizer, 
              ids = id, 
              progressbar = FALSE)
  
  vocab = create_vocabulary(it)
  vocab = prune_vocabulary(vocab, term_count_min = 25,vocab_term_max = 10000)
  
  #Construct a vector space 
  vectorizer = vocab_vectorizer(vocab)
  #Construct Document Term Matrix from the vectorizer
  print("Generating Document Term Matrix")
  dtm = create_dtm(it, vectorizer)
  colnames(dtm) <- paste0("hash_",colnames(dtm))
  return(dtm)
}
remove_high_correlated <- function(data_matrix)
{
  class(data_matrix)<-"numeric"
  correlation_matrix = cor(data_matrix,use = "complete.obs")
  high_correlated = findCorrelation(correlation_matrix, cutoff=0.8) # putt any value as a "cutoff" 
  high_correlated = sort(high_correlated)
  print(high_correlated)
  reduced_data_matrix = data_matrix[,-c(high_correlated)]
  return(reduced_data_matrix)
}
featurize_image <- function(url)
{
  
}