#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import plotly.graph_objects as go 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
import string

import base64
from io import BytesIO

import urllib.request
import json 
import datetime

from wordcloud import WordCloud, STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pprint import pprint
from IPython import display

nltk.download('vader_lexicon')

#Authentication
consumerKey="U4y7XJ0ejIy9ThRje9UtZXl8N"
consumerSecret="M2QyH0E4VpxBwPnXWmV9Bnwy0ZTWobrWuuoFv02PsMECyLq1oJ"
accessToken="1268986499728580610-57f201gaeuOB0yHS4LdHcqm0ekNT6P"
accessTokenSecret="oIWLBcPVBEf5XZ8lQGlBWjC9ARV8PMrgL26mPkspZtYip"

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth, wait_on_rate_limit= True)

#Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#111111',
    'text': 'firebrick'
}

app.layout = html.Div([
    html.H6("Sentiment Analysis",style={
        'textAlign': 'left-center',
        'color': colors['text'],
        'fontWeight': 'bold',
        }),
    dcc.Location(id='url', refresh=False),
    html.Div(id='home_page')
])


index_page = html.Div([
    dcc.Link('Analyze a Twitter Account', href='/twitter'),
    html.Br(),
    dcc.Link('Analyze a Subreddit Forum', href='/reddit'),
])


#Twitter Page
twitter_layout = html.Div([
    html.H6('Twitter'),
    dcc.Input(
        id='twitterhandle',
        placeholder='Enter a Twitter Handle',
        type='text',
        value='',
        style={'width': 300}
    ),
    html.Button('Submit', id='button', n_clicks=0),
    html.Div(id='twitter_output', style={'whiteSpace': 'pre-line'}),
    html.Br(),
    dcc.Link('Analyze using Reddit', href='/reddit'),
    html.Br(),
    dcc.Link('Go back to home', href='/'),
])

@app.callback(dash.dependencies.Output('twitter_output', 'children'),
              [dash.dependencies.Input('button', 'n_clicks')],
              [dash.dependencies.State('twitterhandle', 'value')])
def twitter_page(n_clicks, twitterhandle):
    if n_clicks > 0:
        # Extract 1000 tweets from twitter user
        posts = api.user_timeline(screen_name = twitterhandle, count=10000, lang = "en", tweet_mode="extended")
        
        # Create dataframe 
        df = pd.DataFrame( [tweet.full_text for tweet in posts] , columns=['Tweets'])
        
        # Clean text
        # Create function to clean tweets
        def cleanTxt(text):
            text = re.sub(r'@[A-Za-z0-9]+', '', text) #removes @mentions
            text = re.sub(r'#', '', text) #removes '#'
            text = re.sub(r'RT[\s]+', '', text) # removes RT
            text = re.sub(r'https?:\/\/\S+' , '', text) #removes links
            return text
        
        df['Text']= df['Tweets'].apply(cleanTxt)
        
        # Create function to get the subjectivity
        def getSubjectivity(text):
            return TextBlob(text).sentiment.subjectivity
        
        # Create function to get the polarity
        def getPolarity(text):
            return TextBlob(text).sentiment.polarity
        
        # Create two new columns
        df['Subjectivity'] = df['Text'].apply(getSubjectivity)
        df['Polarity'] = df['Text'].apply(getPolarity)
        
        # Calculate the negative, neutral and positive analysis
        def getAnalysis(score):
            if score < 0:
                return 'Negative'
            elif score == 0:
                return 'Neutral'
            else:
                return 'Positive'
        
        df['Sentiment'] = df['Polarity'].apply(getAnalysis)
        
        #Create Pie Chart
        labels = df['Sentiment'].value_counts().index
        values = df['Sentiment'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        
        
        # Clean the text
        allWords = ' '.join( [twts for twts in df['Text']] )
        allWords_lower = allWords.lower()
        stop = set(stopwords.words('english') + list(string.punctuation))
        new_stopwords = ['...', '``', "''",  '’', "'s", "n't" ]
        new_stopwords_list = stop.union(new_stopwords)

        text_tokens = nltk.word_tokenize(allWords_lower)
        text_no_stop_words_punct = [t for t in text_tokens if t not in new_stopwords_list and t not in string.punctuation]

        filtered_string = (" ").join(text_no_stop_words_punct)
        
        # Convert the long string to a dictionary with frequency counts.
        def word_count(str):
            counts = dict()
            words = str.split()

            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1

            return counts

        twitter_wordcloud = ( word_count(filtered_string))
        
        # Create Word Cloud
        wc = WordCloud().generate_from_frequencies(frequencies=twitter_wordcloud)
        wc_img = wc.to_image()
        with BytesIO() as buffer:
            wc_img.save(buffer, 'png')
            img2 = base64.b64encode(buffer.getvalue()).decode()


        #Display Pie Chart and Word Cloud
        twitter_results = html.Div([ 
            html.Div(
            dcc.Graph(id='graph1', figure=fig)
            ,style={'width': '49%', 'display': 'inline-block'}), 
            html.Div( 
            children=[html.Img(src="data:image/png;base64," + img2,
                            style={'height':'50%', 'width': '50%'})]
             ,style={'width': '49%', 'display': 'block', 'textAlign':'center'}), 
        ])
        
        return twitter_results


#Reddit Page
reddit_layout = html.Div([
    html.H6('Reddit'),
    dcc.Input(
        id='subreddit',
        placeholder='Enter a Subreddit Forum',
        type='text',
        value='',
        style={'width': 300}
    ),
    html.Button('Submit', id='button', n_clicks=0),
    html.Div(id='reddit_output', style={'whiteSpace': 'pre-line'}),
    html.Br(),
    dcc.Link('Analyze using Twitter', href='/twitter'),
    html.Br(),
    dcc.Link('Go back to home', href='/')
])

@app.callback(dash.dependencies.Output('reddit_output', 'children'),
              [dash.dependencies.Input('button', 'n_clicks')],
              [dash.dependencies.State('subreddit', 'value')])
def reddit_radios(n_clicks, subreddit):
    if n_clicks > 0:
        #return 'You have selected "{}"'.format(subreddit)
        
        #Define function to pull up historical Reddit post information
        def load_results(lower_bound_timestamp, upper_bound_timestamp, target_result_size, target_subreddit, score_threshold):
            headline_collection = set()
            
            reddit_data_url = f"https://api.pushshift.io/reddit/submission/search/?after={lower_bound_timestamp}&before={upper_bound_timestamp}&sort_type=score&sort=desc&subreddit={target_subreddit}&limit={target_result_size}&score={score_threshold}"
    
            try:
                with urllib.request.urlopen(reddit_data_url) as url:
                    data = json.loads(url.read().decode())
            
                    for submission in data['data']:
                        headline_collection.add(submission['title'])

                return headline_collection
            except urllib.error.HTTPError as e:
                print(e.__dict__)
                return set()
            except urllib.error.URLError as e:
                print(e.__dict__)
                return set()
            
        # Get Reddit posts
        headlines = set()

        time_now = datetime.datetime.now()

        limit_delta = 7
        limit_lower_delta = 6

        result_size = 1000
        score_limit = ">0"

        for i in range(0, 8):
            previous_timestamp = int((time_now - datetime.timedelta(days=limit_delta)).timestamp())
            current_timestamp = int((time_now - datetime.timedelta(days=limit_lower_delta)).timestamp())

            full_collection = load_results(previous_timestamp, current_timestamp, result_size, subreddit, score_limit)
            headlines = headlines.union(full_collection)
    
            limit_delta = limit_delta - 1
            limit_lower_delta = limit_lower_delta - 1
    
            display.clear_output()
        
        # Calculate polarity to get the sentiment
        sia = SentimentIntensityAnalyzer()
        results = []

        for line in headlines:
            pol_score = sia.polarity_scores(line)
            pol_score['headline'] = line
            results.append(pol_score)
            
        #Convert the results to a dataframe
        df = pd.DataFrame.from_records(results)
        
        #Label the results accordingly
        df['label'] = 'Neutral'
        df.loc[df['compound'] > 0.1, 'label'] = 'Positive'
        df.loc[df['compound'] < -0.1, 'label'] = 'Negative'
        
        #Create Pie Chart
        labels = df['label'].value_counts().index
        values = df['label'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
        reddit_pie = html.Div([
            dcc.Graph(figure=fig)])
        
        # Clean the text
        allWords = ' '.join( [rdf for rdf in df['headline']] )
        allWords_lower = allWords.lower()
        stop = set(stopwords.words('english') + list(string.punctuation))
        new_stopwords = ['...', '``', "''",  '’', "'s", "n't" ]
        new_stopwords_list = stop.union(new_stopwords)

        text_tokens = nltk.word_tokenize(allWords_lower)
        text_no_stop_words_punct = [t for t in text_tokens if t not in new_stopwords_list and t not in string.punctuation]

        filtered_string = (" ").join(text_no_stop_words_punct)
        
        # Convert the long string to a dictionary with frequency counts.
        def word_count(str):
            counts = dict()
            words = str.split()

            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1

            return counts

        reddit_wordcloud = ( word_count(filtered_string))
        
        # Create Word Cloud
        wc = WordCloud().generate_from_frequencies(frequencies=reddit_wordcloud)
        wc_img = wc.to_image()
        with BytesIO() as buffer:
            wc_img.save(buffer, 'png')
            img2 = base64.b64encode(buffer.getvalue()).decode()
        
        #Display Pie Chart and Word Cloud
        reddit_results = html.Div([ 
            html.Div(
            dcc.Graph(id='graph1', figure=fig)
            ,style={'width': '49%', 'display': 'inline-block'}), 
            html.Div( 
            children=[html.Img(src="data:image/png;base64," + img2,
                            style={'height':'50%', 'width': '50%'})]
             ,style={'width': '49%', 'display': 'block', 'textAlign':'center'}), 
        ])
        
        return reddit_results
        

# Update the index
@app.callback(dash.dependencies.Output('home_page', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/twitter':
        return twitter_layout
    elif pathname == '/reddit':
        return reddit_layout
    else:
        return index_page

if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:




