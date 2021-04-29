from src.reddit_handler import RedditHandler
import matplotlib.pyplot as plt
import matplotlib.image
import numpy as np
import pickle
import datetime
import os
import pandas as pd
import statistics


# initializing RedditHandler
out_folder = 'topic_data' 
extract_post = False  # True if you want to extract Post data, False otherwise
extract_comment = True  # True if you want to extract Comment data, False otherwise
post_attributes = ['id', 'author', 'created_utc', 'num_comments', 'over_18', 'is_self', 'score', 'selftext', 'stickied',
                   'subreddit', 'subreddit_id', 'title']  # default
comment_attributes = ['id', 'author', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'subreddit_id', 'body',
                      'score']  # default
my_handler = RedditHandler(out_folder, extract_post, extract_comment, post_attributes=post_attributes,
                           comment_attributes=comment_attributes)

# create, for each category and semester, interaction networks based on the comments retrieved above
categories = ['guncontrol','politics','minority']
semesters = [('01/01/2017','01/07/2017'),('01/07/2017','01/01/2018'),('01/01/2018','01/07/2018'),('01/07/2018','01/01/2019'),('01/01/2019','01/07/2019')]

path_post = r'semester_2kpost'
path_data = r'User_comments'
path_labels = r'Categories_networks'
stats = list()

for topic in categories:
    cnt_semester = 0
    for semester in semesters:
        period0 = datetime.datetime.strptime(semester[0], "%d/%m/%Y").strftime("%Y-%m-%d")
        period1 = datetime.datetime.strptime(semester[1], "%d/%m/%Y").strftime("%Y-%m-%d")
        semester_path_post = os.path.join(path_post, f'{topic}_{period0}_{period1}.pickle') 
        semester_path_data = os.path.join(path_data, topic, f'{topic}_{period0}_{period1}') 
        with open(semester_path_post, "rb") as input_file:
            post_ids_authors = pickle.load(input_file) # loading dict with post_id as key and author name as value
        post_ids_authors_2 = dict()
        for ids in post_ids_authors:
            post_ids_authors_2[ids.split('_')[1]] = post_ids_authors[ids]
        # extracting user data
        start_date = semester[0] 
        end_date = semester[1] 
        path_semester_label = os.path.join(path_labels, topic, f'{topic}_{period0}_{period1}_labels.csv') 
        # create network and apply EVA community Detection
        graph_stats = my_handler.create_network(topic, post_ids_authors_2, semester_path_data, path_semester_label, cnt_semester)
        stats.append(graph_stats)
        cnt_semester += 1

# creating a csv with network statistics for each category, semester
df_stats = pd.DataFrame(data=stats)
df_stats = df_stats.set_index('Topic')
df_stats.to_csv(r'Categories_networks\Topic_Networks_stats.csv')

# creating final echo chamber plot from EVA Community Detection
img1 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/guncontrol_2017-01-01_2017-07-01.png')
img2 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/guncontrol_2017-07-01_2018-01-01.png')
img3 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/guncontrol_2018-01-01_2018-07-01.png')
img4 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/guncontrol_2018-07-01_2019-01-01.png')
img5 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/guncontrol_2019-01-01_2019-07-01.png')
row1 = np.concatenate((img1, img2, img3, img4, img5), axis=1)
img6 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/minority_2017-01-01_2017-07-01.png')
img7 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/minority_2017-07-01_2018-01-01.png')
img8 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/minority_2018-01-01_2018-07-01.png')
img9 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/minority_2018-07-01_2019-01-01.png')
img10 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/minority_2019-01-01_2019-07-01.png')
row2 = np.concatenate((img6, img7, img8, img9, img10), axis=1)
img11 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/politics_2017-01-01_2017-07-01.png')
img12 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/politics_2017-07-01_2018-01-01.png')
img13 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/politics_2018-01-01_2018-07-01.png')
img14 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/politics_2018-07-01_2019-01-01.png')
img15 = matplotlib.image.imread(r'topic_data/Categories_networks/graphs/politics_2019-01-01_2019-07-01.png')
row3 = np.concatenate((img11, img12, img13, img14, img15), axis=1)
new_image = np.concatenate((row1, row2, row3))
matplotlib.image.imsave(r'topic_data/Categories_networks/graphs/plot_tot_ec.png', new_image)
