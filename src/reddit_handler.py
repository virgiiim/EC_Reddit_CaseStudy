import datetime
from datetime import date
from dateutil.relativedelta import relativedelta
import time
import requests
import json
import random
import os
import os.path
import pandas as pd
import shutil
import zipfile
import glob
import pickle
import networkx as nx
#for text cleaning
import string
import re
import pandas as pd
import cdlib
import networkx as nx
# Eva Community Detection
from cdlib.utils import convert_graph_formats
from collections import defaultdict
from cdlib import AttrNodeClustering
from cdlib.utils import convert_graph_formats
from cdlib.algorithms.internal.ILouvain import ML2
from Eva import eva_best_partition, modularity, purity
from cdlib import ensemble
from cdlib import evaluation
from cdlib import viz
from cdlib import algorithms
from collections import Counter
from cdlib.algorithms import eva

import statistics
import matplotlib.pyplot as plt
import operator

class RedditHandler:
    ''' 
    class responsible for extracting and processing reddit data and the creation of users' network
    '''
    def __init__(self, out_folder, extract_post, extract_comment, post_attributes=['id','author', 'created_utc', 'num_comments', 'over_18', 'is_self', 'score', 'selftext', 'stickied', 'subreddit', 'subreddit_id', 'title'], comment_attributes=['id', 'author', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'subreddit_id', 'body', 'score']):
        '''
        Parameters
        ----------
        out_folder : str
            path of the output folder
        extract_post: bool
            True if you want to extract Post data, False otherwise
        extract_comment : bool
            True if you want to extract Comment data, False otherwise
        post_attributes : list, optional
            post's attributes to be selected. The default is ['id','author', 'created_utc', 'num_comments', 'over_18', 'is_self', 'score', 'selftext', 'stickied', 'subreddit', 'subreddit_id', 'title']
        comment_attributes : list, optional
            comment's attributes to be selected. The default is ['id', 'author', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'subreddit_id', 'body', 'score']
        '''

        self.out_folder = out_folder
        if not os.path.exists(self.out_folder):
            os.mkdir(self.out_folder)
        self.extract_post = extract_post
        self.extract_comment = extract_comment
        self.post_attributes = post_attributes 
        self.comment_attributes = comment_attributes 
   
    def _post_request_API_periodical(self, start_date, end_date, subreddit):
        '''
        API REQUEST to pushishift.io/reddit/submission
        returns a list of 1000 dictionaries where each of them is a post 
        '''
        url = 'https://api.pushshift.io/reddit/search/submission?&size=500&after='+str(start_date)+'&before='+str(end_date)+'&subreddit='+str(subreddit)
        try:
            r = requests.get(url) # Response Object
            time.sleep(random.random()*0.02) 
            data = json.loads(r.text) # r.text is a JSON object, converted into dict
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            return self._post_request_API_periodical(start_date, end_date, subreddit)
        return data['data'] # data['data'] contains list of posts   
    
    def _post_request_API_user(self, start_date, end_date, username):
        '''
        API REQUEST to pushishift.io/reddit/submission
        returns a list of 1000 dictionaries where each of them is a post 
        '''
        url = 'https://api.pushshift.io/reddit/search/submission?&size=500&after='+str(start_date)+'&before='+str(end_date)+'&author='+str(username)
        try:
            r = requests.get(url) # Response Object
            time.sleep(random.random()*0.02) 
            data = json.loads(r.text) # r.text is a JSON object, converted into dict
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            return self._post_request_API_user(start_date, end_date, username)
        return data['data'] # data['data'] contains list of posts   


    def _comment_request_API_periodical(self, start_date, end_date, subreddit):
        '''
        API REQUEST to pushishift.io/reddit/comment
        returns a list of 1000 dictionaries where each of them is a comment
        '''
        url = 'https://api.pushshift.io/reddit/search/comment?&size=500&after='+str(start_date)+'&before='+str(end_date)+'&subreddit='+str(subreddit)
        try:
            r = requests.get(url) # Response Object
            #time.sleep(random.random()*0.02) 
            data = json.loads(r.text) # r.text is a JSON object, converted into dict
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            return self._comment_request_API_periodical(start_date, end_date, subreddit)
        return data['data'] # data['data'] contains list of comments  

    def _comment_request_API_user(self, start_date, end_date, username, sub):
        '''
        API REQUEST to pushishift.io/reddit/comment
        returns a list of 1000 dictionaries where each of them is a comment
        '''
        url = 'https://api.pushshift.io/reddit/search/comment?&size=500&after='+str(start_date)+'&before='+str(end_date)+'&author='+str(username)+'&subreddit='+str(sub)
        try:
            r = requests.get(url) # Response Object
            time.sleep(random.random()*0.02) 
            data = json.loads(r.text) # r.text is a JSON object, converted into dict
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            return self._comment_request_API_user(start_date, end_date, username, sub)
        return data['data'] # data['data'] contains list of comments  

    def _comment_request_API_linkid(self, linkid, start_date, end_date):
        linkid = linkid.replace('t3_','')
        '''
        API REQUEST to pushishift.io/reddit/comment
        returns a list of 1000 dictionaries where each of them is a comment
        '''
        #https://api.pushshift.io/reddit/comment/search/?link_id=bc99el&limit=20000
        #'https://api.pushshift.io/reddit/search/comment/?link_id='+str(linkid)
        url = 'https://api.pushshift.io/reddit/search/comment/?link_id='+str(linkid)+'&limit=20000&after='+str(start_date)+'&before='+str(end_date)
        try:
            r = requests.get(url) # Response Object
            #time.sleep(random.random()*0.02) 
            data = json.loads(r.text) # r.text is a JSON object, converted into dict
        except (requests.exceptions.ConnectionError, json.decoder.JSONDecodeError):
            return self._comment_request_API_linkid(linkid, start_date, end_date)
        return data['data'] # data['data'] contains list of comments  

    def _clean_raw_text(self, text):
        '''
        Clean raw post/comment text with standard preprocessing pipeline
        '''
        # Lowercasing text
        text = text.lower()
        # Removing not printable characters 
        text = ''.join(filter(lambda x:x in string.printable, text))
        # Removing XSLT tags
        text = re.sub(r'&lt;/?[a-z]+&gt;', '', text)
        text = text.replace(r'&amp;', 'and')
        text = text.replace(r'&gt;', '') # TODO: try another way to strip xslt tags
        # Removing newline, tabs and special reddit words
        text = text.replace('\n',' ')
        text = text.replace('\t',' ')
        text = text.replace('[deleted]','').replace('[removed]','')
        # Removing URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        # Removing numbers
        text = re.sub(r'\w*\d+\w*', '', text)
        # Removing Punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Removing extra spaces
        text = re.sub(r'\s{2,}', " ", text)
        # Stop words? Emoji?
        return text

    
    def extract_periodical_data(self, start_date, end_date, categories, n_months):
        '''
        extract Reddit data from a list of subreddits in a specific time-period
        Paramters
        ----------
        start_date : str
            beginning date in format %d/%m/%Y
        end_date : str
            end date in format %d/%m/%Y
        categories : dict
            dict with category name as key and list of subreddits in that category as value
        n_months : int
            integer indicating the time period considered, if you don't want it n_months = 0
        '''
        # formatting date
        pretty_start_date = start_date.replace('/','-')
        pretty_end_date = end_date.replace('/','-')
        real_start_date = start_date 
        real_end_date = end_date 
        # converting date from format %d/%m/%Y to UNIX timestamp as requested by API
        start_date = int(time.mktime(datetime.datetime.strptime(start_date, "%d/%m/%Y").timetuple()))
        end_date = int(time.mktime(datetime.datetime.strptime(end_date, "%d/%m/%Y").timetuple()))
        raw_data_folder = os.path.join(self.out_folder, 'Categories_raw_data')
        if not os.path.exists(raw_data_folder):
            os.mkdir(raw_data_folder)
        categories_keys = list(categories.keys())
        i = 0 #to iter over categories keys
        for category in categories.keys():
            print(f'Extracting category: {categories_keys[i]}')
            users = dict() #users with post & comment shared in different subreddit belonging to the same category
            for sub in categories[category]:
                print(f'Extracting subbredit: {sub}')
                current_date_post = start_date
                current_date_comment = start_date
                # handling time-period
                if n_months == 0:
                    period_post = (datetime.datetime.strptime(real_start_date, "%d/%m/%Y"), datetime.datetime.strptime(real_end_date, "%d/%m/%Y"))
                    period_comment = (datetime.datetime.strptime(real_start_date, "%d/%m/%Y"), datetime.datetime.strptime(real_end_date, "%d/%m/%Y"))
                else:
                    end_period = datetime.datetime.strptime(real_start_date, "%d/%m/%Y") + relativedelta(months=+n_months)
                    period_post = (datetime.datetime.strptime(real_start_date, "%d/%m/%Y"), end_period)
                    period_comment = (datetime.datetime.strptime(real_start_date, "%d/%m/%Y"), end_period)
     
                # extracting posts
                if self.extract_post:
                    posts = self._post_request_API_periodical(current_date_post, end_date, sub) #first call to API
                    while len(posts) > 0: #collecting data until there are no more posts to extract in the time period considered
                        # TODO: check if sub exists!
                        for raw_post in posts: 
                            # saving posts for each period
                            current_date = datetime.datetime.utcfromtimestamp(raw_post['created_utc']).strftime("%d/%m/%Y")
                            condition1_post = datetime.datetime.strptime(current_date, "%d/%m/%Y") >= period_post[1]
                            condition2_post = (datetime.datetime.strptime(current_date, "%d/%m/%Y") +  relativedelta(days=+1)) >= datetime.datetime.strptime(real_end_date, "%d/%m/%Y")
                            if condition1_post or condition2_post: 
                                 # Saving data: for each category a folder 
                                path_category = os.path.join(raw_data_folder, f'{categories_keys[i]}_{pretty_start_date}_{pretty_end_date}')
                                if not os.path.exists(path_category):
                                    os.mkdir(path_category)
                                pretty_period0_post = period_post[0].strftime('%d/%m/%Y').replace('/','-')
                                pretty_period1_post = period_post[1].strftime('%d/%m/%Y').replace('/','-')
                                path_period_category = os.path.join(path_category, f'{categories_keys[i]}_{pretty_period0_post}_{pretty_period1_post}')
                                if not os.path.exists(path_period_category):
                                    os.mkdir(path_period_category)
                                # for each user in a period category a json file
                                for user in users: 
                                    user_filename = os.path.join(path_period_category, f'{user}.json')
                                    if os.path.exists(user_filename):
                                        with open(user_filename) as fp:
                                            data = json.loads(fp.read())
                                            data['posts'].extend(users[user]['posts'])
                                        with open(user_filename, 'w') as fp:
                                            json.dump(data, fp, sort_keys=True, indent=4)
                                    else:
                                        with open(user_filename, 'w') as fp:
                                            json.dump(users[user], fp, sort_keys=True, indent=4)
                                users = dict()
                                if condition1_post:
                                    period_post = (period_post[1], period_post[1] + relativedelta(months=+n_months))
                                    print('PERIOD_POST', period_post)
                                elif condition2_post:
                                    break

                            # collecting posts
                            if raw_post['author'] not in ['[deleted]', 'AutoModerator']: # discarding data concerning removed users and moderators
                                user_id = raw_post['author']
                                post = dict() #dict to store posts
                                # adding field category
                                post['category'] = category
                                # adding field date in a readable format
                                post['date'] = datetime.datetime.utcfromtimestamp(raw_post['created_utc']).strftime("%d/%m/%Y")
                                # cleaning body field
                                merged_text = raw_post['title']+' '+raw_post['selftext']
                                post['clean_text'] = self._clean_raw_text(merged_text)
                                # adding field time_period in a readable format
                                if n_months != 0: 
                                    post['time_period'] = (period_post[0].strftime('%d/%m/%Y'), period_post[1].strftime('%d/%m/%Y')) 
                                else:
                                    post['time_period'] = (datetime.datetime.utcfromtimestamp(start_date).strftime("%d/%m/%Y"),datetime.datetime.utcfromtimestamp(end_date).strftime("%d/%m/%Y"))
                                # selecting fields 
                                for attr in self.post_attributes: 
                                    if attr not in raw_post.keys(): #handling missing values
                                        post[attr] = None
                                    elif (attr != 'selftext') and (attr != 'title'): # saving only clean text
                                        post[attr] = raw_post[attr]
                                if len(post['clean_text']) > 2:  # avoiding empty posts
                                    if user_id not in users.keys():
                                        if self.extract_post and self.extract_comment:
                                            users[user_id] = {'posts':[], 'comments':[]}
                                        else:
                                            users[user_id] = {'posts':[]}
                                    users[user_id]['posts'].append(post)
                        current_date_post = posts[-1]['created_utc'] # taking the UNIX timestamp date of the last record extracted
                        posts = self._post_request_API_periodical(current_date_post, end_date, sub) 
                        pretty_current_date_post = datetime.datetime.utcfromtimestamp(current_date_post).strftime('%Y-%m-%d')
                        print(f'Extracted posts until date: {pretty_current_date_post}')

                # extracting comments
                if self.extract_comment:
                    comments = self._comment_request_API_periodical(current_date_comment, end_date, sub) # first call to API
                    while len(comments) > 0: #collecting data until there are no more comments to extract in the time period considered
                        for raw_comment in comments:
                            # saving comments for each period 
                            current_date = datetime.datetime.utcfromtimestamp(raw_comment['created_utc']).strftime("%d/%m/%Y")
                            condition1_comment = datetime.datetime.strptime(current_date, "%d/%m/%Y") >= period_comment[1]
                            condition2_comment = (datetime.datetime.strptime(current_date, "%d/%m/%Y") +  relativedelta(days=+1)) >= datetime.datetime.strptime(real_end_date, "%d/%m/%Y")
                            if condition1_comment or condition2_comment: # saving comment for period
                                 # Saving data: for each category a folder 
                                path_category = os.path.join(raw_data_folder, f'{categories_keys[i]}_{pretty_start_date}_{pretty_end_date}')
                                if not os.path.exists(path_category):
                                    os.mkdir(path_category)
                                pretty_period0_comment = period_comment[0].strftime('%d/%m/%Y').replace('/','-')
                                pretty_period1_comment = period_comment[1].strftime('%d/%m/%Y').replace('/','-')
                                path_period_category = os.path.join(path_category, f'{categories_keys[i]}_{pretty_period0_comment}_{pretty_period1_comment}')
                                if not os.path.exists(path_period_category):
                                    os.mkdir(path_period_category)
                                # for each user in a period category a json file
                                for user in users: 
                                    user_filename = os.path.join(path_period_category, f'{user}.json')
                                    if os.path.exists(user_filename):
                                        with open(user_filename) as fp:
                                            data = json.loads(fp.read())
                                            data['comments'].extend(users[user]['comments'])
                                        with open(user_filename, 'w') as fp:
                                            json.dump(data, fp, sort_keys=True, indent=4)
                                    else:
                                        with open(user_filename, 'w') as fp:
                                            json.dump(users[user], fp, sort_keys=True, indent=4)
                                users = dict()
                                if condition1_comment:
                                    period_comment = (period_comment[1], period_comment[1] + relativedelta(months=+n_months))
                                    print('PERIOD_COMMENT', period_comment)
                                elif condition2_comment:
                                    break

                            # collecting comments
                            if raw_comment['author'] not in ['[deleted]', 'AutoModerator']:
                                user_id = raw_comment['author']
                                comment = dict() # dict to store a comment
                                # adding field category
                                comment['category'] = category
                                # adding field date in a readable format
                                comment['date'] = datetime.datetime.utcfromtimestamp(raw_comment['created_utc']).strftime("%d/%m/%Y")
                                # cleaning body field
                                comment['clean_text'] = self._clean_raw_text(raw_comment['body'])
                                # adding time_period fieldin a readable format
                                if n_months != 0: 
                                    comment['time_period'] = (period_comment[0].strftime('%d/%m/%Y'), period_comment[1].strftime('%d/%m/%Y')) 
                                else:
                                    comment['time_period'] = (datetime.datetime.utcfromtimestamp(start_date).strftime("%d/%m/%Y"),datetime.datetime.utcfromtimestamp(end_date).strftime("%d/%m/%Y"))
                                # selecting fields
                                for attr in self.comment_attributes: 
                                    if attr not in raw_comment.keys(): #handling missing values
                                        comment[attr] = None
                                    elif attr != 'body': # saving only clean text
                                        comment[attr] = raw_comment[attr]
                                if len(comment['clean_text']) > 2: # avoiding empty comments
                                    if user_id not in users.keys():
                                        if self.extract_post and self.extract_comment:
                                            users[user_id] = {'posts':[], 'comments':[]} 
                                        else:
                                            users[user_id] = {'comments':[]} 
                                    users[user_id]['comments'].append(comment)
                        current_date_comment = comments[-1]['created_utc'] # taking the UNIX timestamp date of the last record extracted
                        comments = self._comment_request_API_periodical(current_date_comment, end_date, sub) 
                        pretty_current_date_comment = datetime.datetime.utcfromtimestamp(current_date_comment).strftime('%Y-%m-%d')
                        print(f'Extracted comments until date: {pretty_current_date_comment}')
                print(f'Finished data extraction for subreddit {sub}')
            # zip category folder 
            #shutil.make_archive(path_category, 'zip', path_category) 
            #shutil.rmtree(path_category) 
            print('Done to extract data from category:', categories_keys[i])
            i+=1 #to iter over categories elements
    
    def extract_user_data(self, post_ids_authors, subs, topic_name, start_date=None, end_date=None):
        '''
        extract data (i.e., posts and/or comments) of one or more Reddit users 
        Paramters
        ----------
        users_list : list
            list with Reddit users' usernames 
        start_date : str
            beginning date in format %d/%m/%Y, None if you want start extracting data from Reddit beginning (i.e., 23/06/2005)
        end_date : str
            end date in format %d/%m/%Y, None if you want end extracting data at today date

        '''
        # creating folder to record user activities
        raw_data_folder = os.path.join(self.out_folder, 'User_comments')
        if not os.path.exists(raw_data_folder):
            os.mkdir(raw_data_folder)
        # handling dates (i.e., when starting and ending extract data)
        if start_date == None:
            start_date = '23/06/2005' # start_date = when Reddit was launched
        if end_date == None:
            end_date = date.today()
            end_date = end_date.strftime("%d/%m/%Y") # end_date = current date
        # converting date from format %d/%m/%Y to UNIX timestamp as requested by API
        pretty_start_date = start_date.replace('/','-')
        pretty_end_date = end_date.replace('/','-')
        print('semester:', pretty_start_date,'/', pretty_end_date)
        start_date = int(time.mktime(datetime.datetime.strptime(start_date, "%d/%m/%Y").timetuple()))
        end_date = int(time.mktime(datetime.datetime.strptime(end_date, "%d/%m/%Y").timetuple()))
        users_list = list(set(post_ids_authors.values())) # insert one or more Reddit username
        print('# users', len(users_list))
        post_ids = list(post_ids_authors.keys())
        print('# post', len(post_ids))
        
        for username in users_list:
            users = dict()
            for sub in subs: # TODO DEFINE CATEGORIES        
                #print("Begin data extraction for user:", username)
                current_date_post = start_date
                current_date_comment = start_date
                # extracting posts
                if self.extract_post:
                    posts = self._post_request_API_user(current_date_post, end_date, username) #first call to API # TODO change API
                    while len(posts) > 0: #collecting data until reaching the end_date
                        # TODO: check if sub exists!
                        for raw_post in posts: 
                            user_id = raw_post['author']
                            if user_id not in users.keys():
                                if self.extract_post and self.extract_comment:
                                    users[user_id] = {'posts':[], 'comments':[]}
                                else:
                                    users[user_id] = {'posts':[]}
                            post = dict() #dict to store posts
                            # adding field date in a readable format
                            post['date'] = datetime.datetime.utcfromtimestamp(raw_post['created_utc']).strftime("%d/%m/%Y")
                            # cleaning body field
                            try:
                                merged_text = raw_post['title']+' '+raw_post['selftext']
                            except:
                                merged_text = raw_post['title']
                            post['clean_text'] = self._clean_raw_text(merged_text)
                            # selecting fields 
                            for attr in self.post_attributes: 
                                if attr not in raw_post.keys(): #handling missing values
                                    post[attr] = None
                                elif (attr != 'selftext') and (attr != 'title'): # saving only clean text
                                    post[attr] = raw_post[attr]
                            users[user_id]['posts'].append(post)
                    current_date_post = posts[-1]['created_utc'] # taking the UNIX timestamp date of the last record extracted
                    posts = self._post_request_API_user(current_date_post, end_date, username) 
                    pretty_current_date_post = datetime.datetime.utcfromtimestamp(current_date_post).strftime('%Y-%m-%d')
                    print(f'Extracted posts until date: {pretty_current_date_post}')

                # extracting comments
                if self.extract_comment:
                    comments = self._comment_request_API_user(current_date_comment, end_date, username, sub) # first call to API
                    while len(comments) > 0:
                        for raw_comment in comments: 
                            if raw_comment['link_id'] in post_ids: 
                                print("Begin data extraction for user:", username)
                                user_id = raw_comment['author']
                                if user_id not in users.keys():
                                    if self.extract_post and self.extract_comment:
                                        users[user_id] = {'posts':[], 'comments':[]} 
                                    else:
                                        users[user_id] = {'comments':[]} 
                                comment = dict() # dict to store a comment
                                # adding field date in a readable format
                                comment['date'] = datetime.datetime.utcfromtimestamp(raw_comment['created_utc']).strftime("%d/%m/%Y")
                                # cleaning body field
                                comment['clean_text'] = self._clean_raw_text(raw_comment['body'])
                                # selecting fields
                                for attr in self.comment_attributes: 
                                    if attr not in raw_comment.keys(): #handling missing values
                                        comment[attr] = None
                                    elif attr != 'body': # saving only clean text
                                        comment[attr] = raw_comment[attr]
                                users[user_id]['comments'].append(comment)
                        current_date_comment = comments[-1]['created_utc'] # taking the UNIX timestamp date of the last record extracted
                        comments = self._comment_request_API_user(current_date_comment, end_date, username, sub) 
                        pretty_current_date_comment = datetime.datetime.utcfromtimestamp(current_date_comment).strftime('%Y-%m-%d')
                        print(f'Extracted comments until date: {pretty_current_date_comment}')
                #print('Finish data extraction for user:', username)
            # saving data: for each user a json file
            topic_folder = os.path.join(raw_data_folder, topic_name)
            if not os.path.exists(topic_folder):
                os.mkdir(topic_folder)
            path_semester_category = os.path.join(topic_folder, f'{topic_name}_{pretty_start_date}_{pretty_end_date}')
            if not os.path.exists(path_semester_category):
                os.mkdir(path_semester_category)
            for user in users: 
                user_filename = os.path.join(path_semester_category, f'{user}.json')
                with open(user_filename, 'w') as fp:
                    json.dump(users[user], fp, sort_keys=True, indent=4)
        print('Done to extract data for all selected users', users_list)
    
    def extract_comment_fromid(self, post_ids_authors, topic_name, start_date=None, end_date=None):
        '''
        extract comments of a specific post 
        Paramters
        ----------
        post_ids_authors : dict
           dict with id of post as key and post author as value
        topic_name: str
            name of the category
        start_date : str
            beginning date in format %d/%m/%Y, None if you want start extracting data from Reddit beginning (i.e., 23/06/2005)
        end_date : str
            end date in format %d/%m/%Y, None if you want end extracting data at today date

        '''
        # creating folder to record user activities
        raw_data_folder = os.path.join(self.out_folder, 'User_comments')
        if not os.path.exists(raw_data_folder):
            os.mkdir(raw_data_folder)
        # handling dates (i.e., when starting and ending extract data)
        if start_date == None:
            start_date = '23/06/2005' # start_date = when Reddit was launched
        if end_date == None:
            end_date = date.today()
            end_date = end_date.strftime("%d/%m/%Y") # end_date = current date
        # converting date from format %d/%m/%Y to UNIX timestamp as requested by API
        pretty_start_date = start_date.replace('/','-')
        pretty_end_date = end_date.replace('/','-')
        print('semester:', pretty_start_date,'/', pretty_end_date)
        start_date = int(time.mktime(datetime.datetime.strptime(start_date, "%d/%m/%Y").timetuple()))
        end_date = int(time.mktime(datetime.datetime.strptime(end_date, "%d/%m/%Y").timetuple()))
        users_list = list(set(post_ids_authors.values())) # insert one or more Reddit username
        print('# users', len(users_list))
        post_ids = list(post_ids_authors.keys())
        print('# post', len(post_ids))
        
        users = dict()
        for _id in post_ids:
            current_date_comment = start_date

            # extracting comments
            if self.extract_comment:
                comments = self._comment_request_API_linkid(_id, current_date_comment, end_date) # first call to API
                while len(comments) > 0:
                    for raw_comment in comments:
                        if raw_comment['author'] in users_list: 
                            user_id = raw_comment['author']
                            if user_id not in users.keys():
                                if self.extract_post and self.extract_comment:
                                    users[user_id] = {'posts':[], 'comments':[]} 
                                else:
                                    users[user_id] = {'comments':[]} 
                            comment = dict() # dict to store a comment
                            # adding field date in a readable format
                            comment['date'] = datetime.datetime.utcfromtimestamp(raw_comment['created_utc']).strftime("%d/%m/%Y")
                            # cleaning body field
                            comment['clean_text'] = self._clean_raw_text(raw_comment['body'])
                            # selecting fields
                            for attr in self.comment_attributes: 
                                if attr not in raw_comment.keys(): #handling missing values
                                    comment[attr] = None
                                elif attr != 'body': # saving only clean text
                                    comment[attr] = raw_comment[attr]
                            users[user_id]['comments'].append(comment)
                    current_date_comment = comments[-1]['created_utc'] # taking the UNIX timestamp date of the last record extracted
                    comments = self._comment_request_API_linkid(_id, current_date_comment, end_date)
                    pretty_current_date_comment = datetime.datetime.utcfromtimestamp(current_date_comment).strftime('%Y-%m-%d')
        # saving data: for each user a json file
        topic_folder = os.path.join(raw_data_folder, topic_name)
        if not os.path.exists(topic_folder):
            os.mkdir(topic_folder)
        path_semester_category = os.path.join(topic_folder, f'{topic_name}_{pretty_start_date}_{pretty_end_date}')
        if not os.path.exists(path_semester_category):
            os.mkdir(path_semester_category)
        for user in users: 
            user_filename = os.path.join(path_semester_category, f'{user}.json')
            with open(user_filename, 'w') as fp:
                json.dump(users[user], fp, sort_keys=True, indent=4)
        print('Done to extract data for all selected users', users_list)

    @staticmethod
    def _read_net(filename):
        g = nx.Graph()
        with open(filename) as f:
            f.readline()
            for l in f:
                l = l.split(",")
                g.add_edge(l[0], l[1], weight=int(l[2]))
        return g

    @staticmethod
    def _read_labels(filename):
        node_to_label = {}
        with open(filename) as f:
            f.readline()
            for l in f:
                l = l.rstrip().split(",")
                node_to_label[l[0]] = l[2]
        return node_to_label


    @staticmethod
    def _eva(g, nth, topic_name, semesters, cnt_semester):
        def _preprocess_graph(g):
            comps_list = list(nx.connected_components(g))
            max_len = sorted([[len(el),el] for el in comps_list], reverse=True)
            comp_0 = nx.subgraph(g, max_len[0][1])
            
            mapping = dict(zip(comp_0, range(0, len(comp_0))))
            relabel_comp_0 = nx.relabel_nodes(comp_0, mapping)
            
            inv_map = {v: k for k, v in mapping.items()}
        
            return relabel_comp_0, inv_map
        def comm_purity(labels,size):
            majority_label = list()
            purities = list()
            cnt = 0
            print(len(size))
            for x in size:
                for elem in labels:
                    size2 = sum(list(labels[elem]['leaning'].values()))
                    if x == size2:
                        max_label = max(labels[elem]['leaning'].items(), key=operator.itemgetter(1))[0]
                        purity = labels[elem]['leaning'][max_label]/size2
                        purities.append(purity)
                        majority_label.append(max_label)
            cnt+=1
            return (majority_label[:len(size)],purities[:len(size)])
        path_snapshot = os.path.join('topic_data','Categories_networks','snapshots')
        path_graph = os.path.join('topic_data','Categories_networks','graphs')
        nx.set_node_attributes(g, nth, 'leaning')
        relab_comp_max,mapping = _preprocess_graph(g)
        coms, com_labels = eva_best_partition(relab_comp_max, alpha=0.6)
        coms_to_node = defaultdict(list)
        for n, c in coms.items():
            coms_to_node[c].append(n)
        coms_eva = [list(c) for c in coms_to_node.values()]
        res = AttrNodeClustering(coms_eva, relab_comp_max, "Eva", com_labels, method_parameters={"weight": 'weight', "resolution": 1,
                                                                                "randomize": False, "alpha":0.5})
        # computing snapshot
        users_dict = dict()
        eva_comm = list()
        for com in res.communities:
            each_com = list()
            for node in com:
                each_com.append(mapping[node])
            eva_comm.append(each_com)
        for i in range(len(eva_comm)):
            max_label = max(res.coms_labels[i]['leaning'].items(), key=operator.itemgetter(1))[0]
            for node in eva_comm[i]:
                users_dict[node] = max_label
        final = pd.DataFrame(users_dict.items(), columns=['author', 'com_label'])
        final['snapshot_id'] = cnt_semester
        file_name = f'{topic_name}_{semesters}_snapshot.csv'
        final.to_csv(os.path.join(path_snapshot, file_name))
        # make graph
        eva_densities = evaluation.internal_edge_density(relab_comp_max,res, summary=False)
        average_internal_degre = evaluation.average_internal_degree(relab_comp_max,res, summary=False) #The average internal degree of the community set.
        conductance_eva = evaluation.conductance(relab_comp_max,res,summary=False)#Fraction of total edge volume that points outside the community
        cut_ratio = evaluation.cut_ratio(relab_comp_max,res,summary=False) #Fraction of existing edges (out of all possible edges) leaving the community.
        edge_inside = evaluation.edges_inside(relab_comp_max,res,summary=False)#Number of edges internal to the community.
        expansion = evaluation.expansion(relab_comp_max,res,summary=False)#Number of edges per community node that point outside the cluster.
        modularity = evaluation.erdos_renyi_modularity(relab_comp_max,res,summary=False)#erdos_renyi_modularity
        size_eva = evaluation.size(relab_comp_max,res,summary=False)
        eva_evaluation = pd.DataFrame()
        eva_evaluation['internal_edge_density'] = eva_densities
        eva_evaluation['average_internal_degree'] = average_internal_degre
        eva_evaluation['conductance'] = conductance_eva
        eva_evaluation['cut_ratio'] = cut_ratio
        eva_evaluation['edge_inside'] = edge_inside
        eva_evaluation['expansion'] = expansion
        eva_evaluation['size'] = size_eva
        eva_evaluation['purity'] = comm_purity(res.coms_labels,size_eva)[1]
        eva_evaluation['majority_class'] = comm_purity(res.coms_labels,size_eva)[0]
        eva_correct = eva_evaluation.loc[(eva_evaluation['size']>=25)&(eva_evaluation['conductance']<=0.4)]        
        republican = eva_correct.loc[eva_correct['majority_class']=='protrump']
        democratic = eva_correct.loc[eva_correct['majority_class']=='antitrump']
        neutral = eva_correct.loc[eva_correct['majority_class']=='neutral']
        size_rep = republican['size'].to_list()
        size_rep = [x*7 for x in size_rep]
        size_dem = democratic['size'].to_list()
        size_dem = [x*7 for x in size_dem]
        size_neu = neutral['size'].to_list()
        size_neu = [x*7 for x in size_neu]
        fig = plt.figure(figsize=(8, 12))
        eva = plt.scatter('conductance', 'purity', data=republican, marker='o', color='b',s=size_rep,alpha=.6, label='Pro-trump')
        eva = plt.scatter('conductance', 'purity', data=democratic, marker='o', color='r',s=size_dem,alpha=.6, label='Anti-trump')
        eva = plt.scatter('conductance', 'purity', data=neutral, marker='o', color='g',s=size_neu,alpha=.6, label='Neutral')
        ax = fig.add_subplot(111)
        plt.axhline(y=0.7, color='r', linestyle ='--', alpha = 0.8)
        plt.axvline(x=0.5, color='black', linestyle ='--', alpha = 0.8)
        plt.ylim(0, 1)
        plt.xlim(0,0.5)
        if cnt_semester == 0:
            plt.ylabel("Purity", fontsize=35)
            plt.title('01/2017-07/2017', fontsize=30)
        else:
            plt.ylabel("Purity", fontsize=35 , color='white')
        if cnt_semester == 1:
            plt.title('07/2017-01/2018', fontsize=30)
        if cnt_semester == 2:
            plt.xlabel("Conductance", fontsize=35)
            plt.title('01/2018-07/2018', fontsize=30)
        else:
            plt.xlabel("Conductance", fontsize=35, color='white')
        if cnt_semester == 3:
            plt.title('07/2018-01/2019', fontsize=30)
        if cnt_semester == 4:
            lgnd = plt.legend(loc="upper right", scatterpoints=1, prop={'size': 25},  fontsize=28)
            lgnd.legendHandles[0]._sizes = [300]
            lgnd.legendHandles[1]._sizes = [300]
            lgnd.legendHandles[2]._sizes = [300]
            plt.title('01/2019-07/2019', fontsize=30)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.yticks([0.1,0.3,0.5,0.7,0.9])
        plt.xticks([0.1,0.3,0.5])
        plt.tight_layout()
        file_name = f'{topic_name}_{semesters}.png'
        plt.savefig(os.path.join(path_graph, file_name))

    def create_network(self, topic, post_ids, semester_data, semester_label, cnt_semester):

            user_network_folder = os.path.join(self.out_folder, 'Categories_networks')
            if not os.path.exists(user_network_folder):
                os.mkdir(user_network_folder)
            category_path = os.path.join(user_network_folder,topic)
            if not os.path.exists(category_path):
                os.mkdir(category_path)
            post_to_author = {}
            print(semester_data)
            
            last_folder = os.path.basename(os.path.normpath(semester_data))
            users_files = glob.glob(f"{semester_data}{os.sep}*.json")
            with open(os.path.join(category_path, f"{last_folder}.csv"), "w") as out:
                for user_file in users_files:
                    with open(user_file) as fp:
                        data = json.loads(fp.read())
                        for comments in data['comments']:
                            res = f"{comments['id']},{comments['parent_id'].split('_')[1]},{comments['author']},,{'boh'}\n"
                            post_to_author[comments['id']] = comments['author']
                            out.write(res)
                post_to_author.update(post_ids)

            with open(os.path.join(category_path, f"{last_folder}.csv")) as f:
                interactions = dict()
                for row in f:
                    row = row.split(",")
                    try:
                        tid = post_to_author[row[1]]
                        if row[2] != tid: # remove self loops
                            if (row[2],tid) not in interactions:
                                interactions[(row[2],tid)] = 0
                            interactions[(row[2],tid)]+=1
                    except:
                        pass
                final_inter = dict() 
                for link, weight in interactions.items():
                    link = tuple(sorted(link))
                    if link not in final_inter: # from directed to undirected graph 
                        final_inter[link] = 0
                    final_inter[link] += weight

            with open(os.path.join(category_path, f"{last_folder}_complete.csv"), "w") as out:
                for link,weight in final_inter.items():
                    res = f'{link[0]},{link[1]},{weight}\n'
                    out.write(res)
            os.remove(os.path.join(category_path, f"{last_folder}.csv"))

            # compute network statistics
            g = self._read_net(os.path.join(category_path, f"{last_folder}_complete.csv")) # graph
            nth = self._read_labels(semester_label) # node labels
            node_labels = pd.read_csv(semester_label)
            node_labels = node_labels.loc[node_labels.author.isin(g.nodes())]
            node_labels['snapshot_id'] = cnt_semester
            print(node_labels)
            node_labels.to_csv()
            for val, cnt in node_labels.leaning.value_counts().iteritems():
                if val == 'protrump':
                    cnt_protrump = cnt
                elif val == 'antitrump':
                    cnt_antitrump = cnt
                elif val == 'neutral':
                    cnt_neutral = cnt
            comps = list(nx.connected_components(g)) # get a list of connected components (for decreasing size)
            graph_stats = {
                    'Topic': last_folder.split('_')[0],
                    'Semester': last_folder.split('_')[1]+'_'+last_folder.split('_')[2],
                    '# nodes': g.number_of_nodes(),
                    '# edges': g.number_of_edges(),
                    'Average degree': round(sum(dict(g.degree()).values())/float(len(g)),4),
                    'Diameter': nx.diameter(g.subgraph(comps[0])),
                    'Density': round(nx.density(g),4),
                    'Global Clustering': round(nx.average_clustering(g),4),
                    '# Protrump user': cnt_protrump,
                    '# Antitrump user': cnt_antitrump,
                    '# neutral user:': cnt_neutral}
            # Community Detection
            self._eva(g, nth, last_folder.split('_')[0],last_folder.split('_')[1]+'_'+last_folder.split('_')[2], cnt_semester)
            return graph_stats


if __name__ == '__main__':
    # initializing RedditHandler
    cwd = os.getcwd()
    out_folder = os.path.join(cwd, 'RedditHandler_Outputs')
    out_folder = 'EC_Topics_data'
    extract_post = True # True if you want to extract Post data, False otherwise
    extract_comment = True # True if you want to extract Comment data, False otherwise
    post_attributes = ['id','author', 'created_utc', 'num_comments', 'over_18', 'is_self', 'score', 'selftext', 'stickied', 'subreddit', 'subreddit_id', 'title'] # default 
    comment_attributes = ['id', 'author', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'subreddit_id', 'body', 'score'] # default 
    my_handler = RedditHandler(out_folder, extract_post, extract_comment, post_attributes=post_attributes, comment_attributes=comment_attributes)
    start_date = '01/01/2017'
    end_date = '01/07/2017'
    category = {'guncontrol':['guns','guncontrol', 'antiwar', 'Firearms']}
    n_months = 6  # time_period to consider: if you don't want it n_months = 0
    #my_handler.extract_periodical_data(start_date, end_date, category, n_months)
    #my_handler.create_network(category)
    # extracting user data
    users_list = ['17michela', 'BelleAriel', 'EschewObfuscation10'] # insert one or more Reddit username
    start_date = None # None if you want start extracting from Reddit beginning, otherwise specify a date in format %d/%m/%Y 
    end_date = None # None if you want end extracting at today date, otherwise specify a date in format %d/%m/%Y 
    #my_handler.extract_user_data(users_list, start_date=start_date, end_date=end_date)