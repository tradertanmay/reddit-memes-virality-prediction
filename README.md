# reddit-memes-virality-prediction



## Replicating Results

To replicate the results, please follow the steps below:

1. **Sequential Execution**: Execute the cells in the `memes_download.ipynb` notebook sequentially. Keep the following points in mind:

    a) Ensure that `config.py` and `memes_scrapper.py` are located in the same folder as the `memes_download.ipynb` file.

    b) The PRAW library is required, along with the following Reddit API credentials: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT`. Please ensure these are correctly set up.

    c) Modify the paths in the script to specify where you wish to store the downloaded memes and the associated metadata.

    d) The downloaded memes are stored in the specified directory (replace [xxxx] with the actual path), and the metadata is saved in a file named memes_metadata.csv within this repository.

## Additional Notes

- **Dependencies**: Make sure to install all necessary Python packages by running `pip install -r requirements1.txt`.

- **API Credentials**: For obtaining Reddit API credentials, visit [Reddit's Developer Page](https://www.reddit.com/prefs/apps) and follow the instructions to create an app.


2.
