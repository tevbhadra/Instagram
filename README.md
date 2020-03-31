# InstagramPosts_PopularityPrediction
The aim of the project is to use Social Media(Instagram) data to create marketing campaigns.

Dataset I worked on had Instagram posts Url's and some basic features extracted from them. 

Steps Involved in the Project:

1) Extract Image URL's in the Instagram posts using Beautiful Soup.

2) Use Google Cloud Vision API to generate Labels for every image with corresponding scores for each label. [File Name: scrapping_image_features.py]

3) Generate an Image feature matrix by selecting the important labels in the data. [File Name: generate_img_feature_matrix.py]

4) Extract Textual features in the data by generating TFIDF vectors, Document Term matrices on the Instagram Captions and Hashtags data. [File Name: featurize.R]

5) Generate Prediction Labels based on the number of likes, comments for the posts. [File Name: labelling.R]

6) Implement the predictive models. [File Name: QCRI_Modelling_RandomForest_h2o.py, QCRI_Modelling_RandomForest_tuning.py, QCRI_Modelling_XGB_tuning.py, DeepLearning.py]
