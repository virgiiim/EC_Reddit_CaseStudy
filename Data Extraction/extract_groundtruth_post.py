from src.reddit_handler import RedditHandler

# initializing RedditHandler
out_folder = 'Data'
extract_post = True  # True if you want to extract Post data, False otherwise
extract_comment = False  # True if you want to extract Comment data, False otherwise
post_attributes = ['id', 'author', 'created_utc', 'num_comments', 'over_18', 'is_self', 'score', 'selftext', 'stickied',
                   'subreddit', 'subreddit_id', 'title']  # default
comment_attributes = ['id', 'author', 'created_utc', 'link_id', 'parent_id', 'subreddit', 'subreddit_id', 'body',
                      'score']  # default
my_handler = RedditHandler(out_folder, extract_post, extract_comment, post_attributes=post_attributes,
                           comment_attributes=comment_attributes)

# extracting periodical data
start_date = '01/01/2017'
end_date = '01/07/2019'
category = {'protrump': ['The_Donald'], 'antitrump':['Fuckthealtright', 'EnoughTrumpSpam']}
n_months = 6  # time_period to consider: if you don't want it n_months = 0
my_handler.extract_periodical_data(start_date, end_date, category, n_months)

