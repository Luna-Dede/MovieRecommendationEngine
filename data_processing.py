#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:09:41 2023

@author: huangrunzhe
"""
import pandas as pd

## Data processing

# Approximate required data:
# Movie title, id, Category, year, rating, number of ratings
## Read metadata
movie_meta_file = "movies_metadata.csv"
df_meta = pd.read_csv(movie_meta_file)
#df_meta = df_meta.set_index(df_meta['id'].str.strip().replace(',','').astype(int))

# Extract important features
meta_features = ['genres', 'imdb_id', 'original_language', 'revenue', 'release_date', 'spoken_languages', 'title', 'vote_average', 'vote_count', 'overview']
data = df_meta[meta_features]
#data=data.dropna(axis= 0)

## Handles genres type name
genres_all=[]
data['genres_split']=data['genres']
for i in range(len(data['genres'])):
    data['genres'][i]=eval(data['genres'][i])
    a=[]
    for j in range(len(data['genres'][i])):
        a.append(data['genres'][i][j]['name'])
        if j==0 and not data['genres'][i][j]['name'] is None:
            data['genres_split'][i]=data['genres'][i][j]['name']
        if data['genres'][i][j]['name'] not in genres_all:
            genres_all.append(data['genres'][i][j]['name'])
    data['genres'][i]=a

# Process time data
for i in range(len(data['release_date'])):
    if not pd.isna(data['release_date'][i]):
        data['release_date'][i]=data['release_date'][i].split('-',1)[0]

# Process range data
data['scale']=data['vote_average']
for i in range(len(data['vote_average'])):
    if data['vote_average'][i]<=1:
        data['scale'][i]='0-1'
    elif data['vote_average'][i]<=2:
        data['scale'][i]='1-2'
    elif data['vote_average'][i]<=3:
        data['scale'][i]='2-3'
    elif data['vote_average'][i]<=4:
        data['scale'][i]='3-4'
    elif data['vote_average'][i]<=5:
        data['scale'][i]='4-5'
    elif data['vote_average'][i]<=6:
        data['scale'][i]='5-6'
    elif data['vote_average'][i]<=7:
        data['scale'][i]='6-7'
    elif data['vote_average'][i]<=8:
        data['scale'][i]='7-8'
    elif data['vote_average'][i]<=9:
        data['scale'][i]='8-9'
    elif data['vote_average'][i]<=10:
        data['scale'][i]='9-10'
    else:
        data['scale'][i]='others'   

# Delete useless data
for i in range(len(data['genres_split'])):
    if data['genres_split'][i]=='[]':
        data=data.drop([i],axis=0)

# Delete missing value
data=data.dropna(axis=0)

# # Count the number of movie voters each year
# df = data
# df['vote_count'] = df['vote_count'].astype('int')
# type_visit_mean = pd.DataFrame(df['vote_count'].groupby(df['release_date']).agg('mean'))

# # Calculate the average movie vote score
# df['vote_average'] = df['vote_average'].astype('float')
# type_score_mean = pd.DataFrame(df['vote_average'].groupby(df['release_date']).agg('mean').round(2))

# Take the unique year and get the maximum and minimum values
years = data['release_date'].unique()
max_year = int(max(years))
min_year = int(min(years))

# Get the unique genre list
genres_list = genres_all