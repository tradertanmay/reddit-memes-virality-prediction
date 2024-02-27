# reddit-memes-virality-prediction



## Replicating Results

To replicate the results, please follow the steps below:

1. **Sequential Execution**: Execute the cells in the `memes_download.ipynb` notebook sequentially. Keep the following points in mind:

    a) Ensure that `config.py` , `memes_scrapper.py` and `preprocessing.py`are located in the same folder as the `memes_download.ipynb` file.

    b) The PRAW library is required, along with the following Reddit API credentials: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, and `REDDIT_USER_AGENT`. Please ensure these are correctly set up.

    c) Modify the paths in the script to specify where you wish to store the downloaded memes and the associated metadata.

    d) The downloaded memes are stored in the specified directory (replace [xxxx] with the actual path), and the metadata is saved in a file named memes_metadata.csv within this repository.

## Additional Notes

- **Dependencies**: Make sure to install all necessary Python packages by running `pip install -r requirements1.txt`.

- **API Credentials**: For obtaining Reddit API credentials, visit [Reddit's Developer Page](https://www.reddit.com/prefs/apps) and follow the instructions to create an app.


2. **Dataset Preparation and Analysis**:
   
    a) **Extended Memes Dataset for Replication**: To facilitate the replication of this study beyond the two-week limit imposed by Reddit, all memes used in this study have been uploaded to an external platform. This ensures that future researchers can download these memes for their analyses. I have uploaded all the downloaded memes to this website for replication.
   
    b) **Metadata Storage**: The metadata associated with these memes is stored in this repository as `memes_metadata.csv`. This file can be used for initial analysis and understanding the dataset context.
   
    c) **Feature Addition to Metadata**: To add features to the memes' metadata, execute the cells in the `metadata_with_features.ipynb` notebook sequentially. The enhanced metadata will be saved in a file named `metadata_with_features.csv`, consisting of the original metadata with additional features included.
   d) Ensure that memes_text_image.py are located in the same folder as the  `metadata_with_features.ipynb`file.

By following these steps, researchers can replicate the study and extend the analysis with the provided dataset.

