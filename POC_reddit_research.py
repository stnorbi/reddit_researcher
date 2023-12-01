import praw
import os
from dotenv import load_dotenv
import datetime
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import spacy
import time

load_dotenv()

# Load spaCy's English language model
nltk.download('stopwords')
nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")

reddit = praw.Reddit(client_id=os.getenv('CLIENT_ID'),
                   client_secret=os.getenv('CLIENT_SECRET'),
                   user_agent=os.getenv('USER_AGENT'))

corpus=[]
# Fetching submissions in chunks
subreddit_name = 'datasets'
submissions = []
num_submissions_to_fetch = 10000
chunk_size = 1000  # Adjust the chunk size based on API limitations
after = None


# Példa: Subreddit adatainak lekérése
#subreddit = reddit.subreddit('datasets')
while len(submissions) < num_submissions_to_fetch:
        # Set the 'after' parameter to continue from the last fetched submission
    submissions_batch = list(reddit.subreddit(subreddit_name).new(limit=chunk_size, params={'after': after}))
    
    # Check if there are no more submissions
    if not submissions_batch:
        break

    submissions.extend(submissions_batch)
    
    # Update the 'after' parameter to continue from the last submission in the batch
    print(submissions_batch[-1].title)
    after = submissions_batch[-1].title

    print(f"Fetched {len(submissions)} submissions")

    #time.sleep(2)

for submission in submissions:
    doc = nlp(submission.selftext)
    dataset_names = [token.text for token in doc if token.pos_ == "PROPN" and token.is_alpha]


    corpus.append({
            'ID' : submission.id,
            'Title': submission.title,
            'Text': ' '.join(dataset_names),
            'Score': submission.score,
            'URL': submission.url,
            'Number of Comments': submission.num_comments,
            'Created UTC': datetime.datetime.fromtimestamp(submission.created_utc),
        })

# Creating a Pandas DataFrame from the list of post data
df = pd.DataFrame(corpus)

# Displaying the DataFrame
#print(df.head())


# Function to preprocess text (remove stopwords, lowercase, etc.)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]
    return ' '.join(words)

# Function to remove specific words from text
def remove_specific_words(text, words_to_remove):
    for word in words_to_remove:
        text = text.replace(word, '')
    return text


# Apply text preprocessing to the 'Text' column
df['ProcessedText'] = df['Text'].apply(preprocess_text)
df['ProcessedTitle'] = df['Title'].apply(preprocess_text)

blacklist=[
            "data",
            "dataset",
            "https",
            "json",
            "available",
            "find",
            "kaggle",
            "database",
            "information",
            "comment",
            "new",
            "source",
           ]

# Apply the remove_specific_words function to the 'Text' column

df['ProcessedText'] =df['ProcessedText'].apply(lambda x: remove_specific_words(x, blacklist))

# Combine all processed text into a single string
#all_text = ' '.join(df['ProcessedText'].values)
all_text = ' '.join(df['ProcessedTitle'].values)
#print(all_text)




# Generate WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

# Displaying the DataFrame
print(df.head())

with pd.ExcelWriter('reddit_src_data.xlsx') as writer:  
        df.to_excel(writer, sheet_name='REDDIT_POSTS')

# Plot the WordCloud image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

