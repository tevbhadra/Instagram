# Code implemented by Eswara Veerabhadra
import pandas as pd
from google.cloud import vision
import os
import logging

ww4 = pd.read_csv('C:/Users/tevbh/Downloads/x4.csv')
ww4.shape
ww4= ww4[112001:112005]
ww4.to_csv('C:/Users/tevbh/Downloads/test.csv')


os.chdir('C:/Users/tevbh/Downloads/Capstone Project/Images')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:/Users/tevbh/Downloads/QCRI/My First Project-fd1079270cc2.json"


#%%
def detect_labels_uri(uri):
    try:
        """Detects labels in the file located in Google Cloud Storage or on the
        Web."""
        # Instantiates a client
        client = vision.ImageAnnotatorClient()
        image = vision.types.Image()
        image.source.image_uri = uri

        response = client.label_detection(image=image)
        labels = response.label_annotations
        z = []
        y = []
        for label in labels:
             y.append((label.description))
             z.append(label.score)
        print('Running')
        client.transport.channel.close()
        return y,z
    except:
        return None

ww4['Labels']= ww4.image_url.apply(detect_labels_uri)

