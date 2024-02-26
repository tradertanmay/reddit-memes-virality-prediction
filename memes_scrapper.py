import os
import requests
import pandas as pd
import praw
import urllib.parse
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)
import os
import requests
import pandas as pd
from datetime import datetime
import urllib.parse

def download_subreddit_images(subreddit_name, num_posts=10000, download_path="downloads"):
    """
    Download images from a specified subreddit.

    :param subreddit_name: Name of the subreddit to download images from.
    :param num_posts: Number of posts to download images from. Default is 10.
    :param download_path: Base path to save the downloaded images. Default is "downloads".
    """
    results = []
    download_directory = os.path.join(download_path, subreddit_name)  # Organized download directory
    os.makedirs(download_directory, exist_ok=True)

    for submission in reddit.subreddit(subreddit_name).hot(limit=num_posts):
        if submission.url.endswith(('.jpg', '.jpeg', '.png')):
            image_url = submission.url
            filename = os.path.basename(urllib.parse.urlparse(image_url).path)
            image_path = os.path.join(download_directory, filename)

            if not os.path.exists(image_path):  # Check if file already exists to avoid duplicates
                try:
                    response = requests.get(image_url, stream=True)
                    if response.status_code == 200:
                        with open(image_path, 'wb') as f:
                            for chunk in response.iter_content(8192):
                                f.write(chunk)
                        print(f"Downloaded {filename} from {subreddit_name}")
                    else:
                        print(f"Failed to download {filename}: {response.status_code}")
                except Exception as e:
                    print(f"Error downloading {filename}: {e}")

            has_thumbnail = submission.thumbnail != 'default'

            thumbnail_height = submission.preview['images'][0]['source']['height'] if has_thumbnail else None
            thumbnail_width = submission.preview['images'][0]['source']['width'] if has_thumbnail else None


            # Collecting additional metadata for analysis
            result_dict = {
                'Subreddit': subreddit_name,
                'Title': submission.title,
                'Upvotes': submission.score,
                'UTC': submission.created_utc,
                'Image URL': image_url,
                'Filename': filename,
                'NSFW': submission.over_18,
                'Num Comments': submission.num_comments,
                'Author': str(submission.author),
                'Submission ID': submission.id,
                'URL Domain': urllib.parse.urlparse(submission.url).netloc,
                'Upvote Ratio': submission.upvote_ratio,
                'Post Flair': submission.link_flair_text,
                'Post Permalink': submission.permalink,
                'Submission Text': submission.selftext,
                'Thumbnail Height': thumbnail_height,
                'Thumbnail Width': thumbnail_width,
                'Has Thumbnail': has_thumbnail
            }
            results.append(result_dict)

    # Saving metadata to a CSV file
    df = pd.DataFrame(results)
    df['Submission Date'] = pd.to_datetime(df['UTC'], unit='s')
    csv_path = os.path.join(download_directory, f"{subreddit_name}_metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")


if __name__ == "__main__":
    subreddits = [
        'wholesomememes', 'dankmemes', 'wholesomemes', 'memes', 'me_irl', 'dank_meme',
        'surrealmemes', 'JustUnsubbed', 'terriblefacebookmemes', 'HistoryMemes',
        'ProgrammerHumor', 'PrequelMemes', 'AdviceAnimals', 'ComedyCemetery', 
        'DeepFried', 'PoliticalHumor', 'WhitePeopleTwitter', 'blackpeopletwitter',
        'sadcringe', '2meirl4meirl', 'absolutelynotme_irl', 'bonehurtingjuice',
        'woooosh', 'lostredditors', 'AntiMeme', 'ExpectationVsReality',
        'interestingasfuck', 'oddlysatisfying', 'youdontsurf'
    ]
    for subreddit in subreddits:
        download_subreddit_images(subreddit)
