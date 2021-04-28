from src.reddit_handler import RedditHandler
import pickle
import datetime
import os

# initializing RedditHandler
out_folder = 'Data'
extract_post = False  # True if you want to extract Post data, False otherwise
extract_comment = True  # True if you want to extract Comment data, False otherwise
post_attributes = ['id', 'author', 'created_utc', 'num_comments', 'over_18', 'is_self', 'score', 'selftext', 'stickied',
                   'subreddit', 'subreddit_id', 'title']  # default
comment_attributes = ['id', 'author', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'subreddit_id', 'body',
                      'score']  # default
my_handler = RedditHandler(out_folder, extract_post, extract_comment, post_attributes=post_attributes,
                           comment_attributes=comment_attributes)

# extracting posts' comments for each category and semester
categories = [{'guncontrol':['Firearms', 'antiwar', 'guncontrol', 'gunpolitics', 'guns','liberalgunowners']}, {'politics': ['Conservative', 'Libertarian', 'democrats', 'Republican','esist','MarchAgainstTrump']},  {'minority': ['Anarchism', 'MensRights', 'AgainstHateSubreddits', 'racism', 'metacanada', 'KotakuInAction'}]
semesters = [('01/01/2017','01/07/2017'),('01/07/2017','01/01/2018'),('01/01/2018','01/07/2018'),('01/07/2018','01/01/2019'),('01/01/2019','01/07/2019')]
path = r'Data/semester' 

for cat in categories:
    for topic in cat:
        for semester in semesters:
            period0 = datetime.datetime.strptime(semester[0], "%d/%m/%Y").strftime("%Y-%m-%d")
            period1 = datetime.datetime.strptime(semester[1], "%d/%m/%Y").strftime("%Y-%m-%d")
            semester_path = os.path.join(path, f'{topic}_{period0}_{period1}.pickle')
            print(semester_path)
            with open(semester_path, "rb") as input_file:
                post_ids_authors = pickle.load(input_file) # loading dict with post_id as key and author name as value
            # extracting user data
            start_date = semester[0]
            end_date = semester[1]
            my_handler.extract_comment_fromid(post_ids_authors, cat[topic], start_date=start_date, end_date=end_date)

