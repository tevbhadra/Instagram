#!/usr/bin/env python
import metadata_parser
import os
import pandas
os.chdir('/home/guu8/qcri_instagram/src/python')
#%%
def get_image_url(final_url):
    try:
        r = metadata_parser.MetadataParser(url= final_url, allow_localhosts = False).get_metadata_link('image')
        return r
    except:
        return print("Something went wrong")
#%%
print("Loading Instagram Data")
data = pandas.read_csv("~/qcri_instagram/data/instagram_240K.csv")
#%%
#data = data.head(100)
print("Getting the Instagram Post URL")
data['post_url'] = 'https://www.instagram.com/p/' + data['url'].astype(str) + '/'
#%%
print('Getting the Instagram Image URL')
#data["image_url"] = data['post_url'].apply(get_image_url)
image_url = []
for i in range(data.shape[0]):
    print(i)
    url = get_image_url(data['post_url'][i])
    image_url.append(url)
print('Writing data with image urls to a file')
data['image_url'] = image_url
data.to_csv(r'../../data/instagram_data_with_urls.csv')
