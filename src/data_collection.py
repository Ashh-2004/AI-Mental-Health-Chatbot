# data_collection.py
import requests
import json
import os
import time
import logging
from typing import List, Dict, Any
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MentalHealthDataCollector:
    def __init__(self, config_path="../config/settings.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = "../data/raw"
        os.makedirs(self.output_dir, exist_ok=True)

    def collect_reddit_data(self, subreddits: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        """
        Collect posts and comments from mental health-related subreddits
        using Reddit API (requires Reddit API credentials).
        """
        import praw  # Install with `pip install praw`

        # Replace these with your Reddit API credentials
        reddit_client_id = "REDDIT_CLIENT_ID"
        reddit_client_secret = "REDDIT_CLIENT_SECRET"
        reddit_user_agent = "REDDIT_USER_AGENT"

        reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent,
        )

        collected_data = []

        for subreddit_name in subreddits:
            logger.info(f"Collecting data from subreddit: {subreddit_name}")
            subreddit = reddit.subreddit(subreddit_name)

            # Fetch posts
            for post in subreddit.hot(limit=limit):
                post_data = {
                    "title": post.title,
                    "selftext": post.selftext,
                    "comments": [],
                }

                # Fetch comments
                post.comments.replace_more(limit=0)
                for comment in post.comments.list():
                    post_data["comments"].append(comment.body)

                collected_data.append(post_data)

                # Log progress
                logger.info(f"Collected post: {post.title}")

        return collected_data

    def save_data(self, data: List[Dict[str, Any]], filename: str):
        """Save collected data to a JSON file."""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Data saved to {output_path}")

    def run(self):
        """Run the data collection process."""
        subreddits = [
            "depression",
            "Anxiety",
            "mentalhealth",
            "SuicideWatch",
            "offmychest",
        ]
        limit_per_subreddit = 50

        # Collect Reddit data
        reddit_data = self.collect_reddit_data(subreddits, limit_per_subreddit)

        # Save the collected data
        timestamp = int(time.time())
        self.save_data(reddit_data, f"reddit_mental_health_{timestamp}.json")


if __name__ == "__main__":
    collector = MentalHealthDataCollector()
    collector.run()