import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from webcolors import hex_to_rgb, CSS3_HEX_TO_NAMES
from transformers import pipeline



import pandas as pd
import cv2
import torch
import pandas as pd
import cv2
import torch
from fer import FER



# Define a function to detect objects using YOLOv5 and recognize facial expressions
def detect_objects_and_emotions(model,emotion_model,image_path):

    image = cv2.imread(image_path)
    
    # Run YOLOv5 model inference
    results = model(image)
    
    # Extract labels from YOLOv5 model output
    detected_objects = [obj[-1] for obj in results.pandas().xyxy[0].values]
    
    # Recognize facial expression
    emotion, _ = emotion_model.top_emotion(image)
    
    return {
        'Detected Objects': ", ".join(detected_objects),
        'Facial Expression': emotion
    }


import os
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from webcolors import hex_to_rgb, CSS3_HEX_TO_NAMES, rgb_to_hex

def process_images3(df, image_folder, output_csv):
    # Make sure 'Filename' is the correct column name containing image file names
    filenames = df['Filename'].tolist()

    def fetch_image_from_file(filename):
        image_path = os.path.join(image_folder, filename)
        return cv2.imread(image_path)

    def color_difference(color1, color2):
        return sum((a - b) ** 2 for a, b in zip(color1, color2))

    def find_closest_color_name(rgb_value):
        closest_color = min(CSS3_HEX_TO_NAMES, key=lambda name: color_difference(rgb_value, hex_to_rgb(name)))
        return CSS3_HEX_TO_NAMES.get(closest_color, "Unknown")

    # Inside the extract_color_features function
    def extract_color_features(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = img.reshape((-1, 3))

        # Calculate average RGB values
        average_red = np.mean(img_array[:, 0])
        average_green = np.mean(img_array[:, 1])
        average_blue = np.mean(img_array[:, 2])

        # Convert the image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Calculate average HSV values
        average_hue = np.mean(img_hsv[:, :, 0])
        average_saturation = np.mean(img_hsv[:, :, 1])
        average_value = np.mean(img_hsv[:, :, 2])

        # Get the dominant color using KMeans clustering
        kmeans = KMeans(n_clusters=1)
        kmeans.fit(img_array)
        top_color = kmeans.cluster_centers_[0].astype(int)

        # Map RGB value to the closest color name
        top_color_name = find_closest_color_name(top_color)

        # Convert RGB to Hex
        hex_color = rgb_to_hex((int(average_red), int(average_green), int(average_blue)))

        return top_color, top_color_name, average_hue, average_saturation, average_value, average_red, average_green, average_blue, hex_color, img_hsv

    # Create lists to store the results
    top_colors_list = []
    top_color_names_list = []
    average_hue_list = []
    average_saturation_list = []
    average_value_list = []
    average_red_list = []
    average_green_list = []
    average_blue_list = []
    hex_color_list = []
    hue_list = []
    saturation_list = []
    value_list = []
    red_list = []
    green_list = []
    blue_list = []

    # Loop through each filename in the DataFrame
    for filename in filenames:
        try:
            img = fetch_image_from_file(filename)

            # Skip small or invalid images
            if img.shape[0] < 10 or img.shape[1] < 10:
                # Append None to maintain alignment
                top_colors_list.append(None)
                top_color_names_list.append(None)
                average_hue_list.append(None)
                average_saturation_list.append(None)
                average_value_list.append(None)
                average_red_list.append(None)
                average_green_list.append(None)
                average_blue_list.append(None)
                hex_color_list.append(None)
                hue_list.append(None)
                saturation_list.append(None)
                value_list.append(None)
                red_list.append(None)
                green_list.append(None)
                blue_list.append(None)
                continue

            # Extract color features
            #print(extract_color_features(img))
            top_color, top_color_name, avg_hue, avg_saturation, avg_value, avg_red, avg_green, avg_blue, hex_color,img_hsv = extract_color_features(img)
               
            # Append results to lists
            top_colors_list.append(top_color)
            top_color_names_list.append(top_color_name)
            average_hue_list.append(avg_hue)
            average_saturation_list.append(avg_saturation)
            average_value_list.append(avg_value)
            average_red_list.append(avg_red)
            average_green_list.append(avg_green)
            average_blue_list.append(avg_blue)
            hex_color_list.append(hex_color)
            
            # Individual color features
            hue_list.append(img_hsv[0, 0, 0])
            saturation_list.append(img_hsv[0, 0, 1])
            value_list.append(img_hsv[0, 0, 2])
            red_list.append(img[0, 0, 0])
            green_list.append(img[0, 0, 1])
            blue_list.append(img[0, 0, 2])

        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            # Append None in case of an error
            top_colors_list.append(None)
            top_color_names_list.append(None)
            average_hue_list.append(None)
            average_saturation_list.append(None)
            average_value_list.append(None)
            average_red_list.append(None)
            average_green_list.append(None)
            average_blue_list.append(None)
            hex_color_list.append(None)
            hue_list.append(None)
            saturation_list.append(None)
            value_list.append(None)
            red_list.append(None)
            green_list.append(None)
            blue_list.append(None)

    # Add the lists to the DataFrame
    df['Top_Color'] = top_colors_list
    df['Top_Color_Name'] = top_color_names_list
    df['Average_Hue'] = average_hue_list
    df['Average_Saturation'] = average_saturation_list
    df['Average_Value'] = average_value_list
    df['Average_Red'] = average_red_list
    df['Average_Green'] = average_green_list
    df['Average_Blue'] = average_blue_list
    df['Hex_Color'] = hex_color_list
    df['Hue'] = hue_list
    df['Saturation'] = saturation_list
    df['Value'] = value_list
    df['Red'] = red_list
    df['Green'] = green_list
    df['Blue'] = blue_list

    # Save the modified DataFrame to the specified CSV file
    df.to_csv(output_csv, index=False)

# Replace 'path/to/images' with the actual path to your image folder






import os
import cv2
import easyocr
import pandas as pd

# Function to extract text from an image using EasyOCR
def extract_text_from_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use EasyOCR to perform text extraction
    reader = easyocr.Reader(['en'])
    result = reader.readtext(gray_image)
    
    # Extracted text from the result
    extracted_text = ' '.join([text[1] for text in result])
    
    return extracted_text

# Function to process DataFrame and extract text from images
def process_dataframe(df, image_folder):
    # Initialize an empty list to store the extracted text
    extracted_texts = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Construct the path to the image file
        image_path = os.path.join(image_folder, row['Filename'])
        
        # Check if the image file exists
        if os.path.exists(image_path):
            # Extract text from the image
            extracted_text = extract_text_from_image(image_path)
        else:
            # If the image file does not exist, set extracted text to None
            extracted_text = None
        
        # Append the extracted text to the list
        extracted_texts.append(extracted_text)
    
    # Add the extracted text to the DataFrame as a new column
    df['Extracted_Text'] = extracted_texts
    
    return df





def classify_sentiment(text):
    # Split the text into chunks of 512 tokens
    chunks = [text[i:i+512] for i in range(0, len(text), 512)]
    
    # Classify sentiment for each chunk
    results = [sentiment_analyzer(chunk) for chunk in chunks]
    
    # Get the overall sentiment based on the majority vote
    labels = [result[0]['label'] for result in results if result]  # Exclude empty results
    if labels:
        positive_count = labels.count('POSITIVE')
        negative_count = labels.count('NEGATIVE')
        neutral_count = labels.count('NEUTRAL')
        
        # Determine the majority sentiment
        if positive_count > negative_count and positive_count > neutral_count:
            return 'POSITIVE'
        elif negative_count > positive_count and negative_count > neutral_count:
            return 'NEGATIVE'
        else:
            return 'NEUTRAL'
    else:
        return 'Unknown'
 

# Function to determine object count
def determine_object_count(objects):
    # Check for NaN, None, empty string, or only whitespace
    if pd.isna(objects) or objects == '' or (isinstance(objects, str) and objects.isspace()):
        return 'no object'
    else:
        object_list = objects.split(', ')
        object_count = len(object_list)
        if object_count > 1:
            return 'multiple'
        else:
            return 'single'






# Sample DataFrame


# Mapping of objects to categories
object_categories = {
    'person': 'human',
    'tie': 'clothing',
    'chair': 'furniture',
    'dog': 'animal',
    'bottle': 'container',
    'car': 'vehicle',
    'cup': 'container',
    'bird': 'animal',
    'clock': 'device',
    'teddy bear': 'toy',
    'book': 'object',
    'cell phone': 'device',
    'sports ball': 'ball',
    'cat': 'animal',
    'tv': 'device',
    'cake': 'food',
    'bowl': 'container',
    'laptop': 'device',
    'kite': 'toy',
    'potted plant': 'plant',
    'vase': 'container',
    'bed': 'furniture',
    'umbrella': 'device',
    'horse': 'animal',
    'dining table': 'furniture',
    'truck': 'vehicle',
    'handbag': 'accessory',
    'donut': 'food',
    'bear': 'animal',
    'wine glass': 'container',
    'traffic light': 'device',
    'motorcycle': 'vehicle',
    'mouse': 'animal',
    'suitcase': 'container',
    'backpack': 'container',
    'bench': 'furniture',
    'frisbee': 'toy',
    'remote': 'device',
    'sheep': 'animal',
    'airplane': 'vehicle',
    'surfboard': 'toy',
    'couch': 'furniture',
    'toilet': 'furniture',
    'baseball bat': 'sports equipment',
    'keyboard': 'device',
    'boat': 'vehicle',
    'scissors': 'tool',
    'bicycle': 'vehicle',
    'cow': 'animal',
    'toothbrush': 'tool',
    'snowboard': 'sports equipment',
    'spoon': 'tool',
    'skateboard': 'toy',
    'refrigerator': 'appliance',
    'bus': 'vehicle',
    'tennis racket': 'sports equipment',
    'train': 'vehicle',
    'parking meter': 'device',
    'broccoli': 'food',
    'fire hydrant': 'device',
    'apple': 'food',
    'baseball glove': 'sports equipment',
    'stop sign': 'device',
    'knife': 'tool',
    'oven': 'appliance',
    'sink': 'appliance',
    'elephant': 'animal',
    'microwave': 'appliance',
    'banana': 'food',
    'skis': 'sports equipment',
    'fork': 'tool',
    'hot dog': 'food',
    'orange': 'food',
    'zebra': 'animal',
    'pizza': 'food',
    'carrot': 'food',
    'sandwich': 'food',
    'giraffe': 'animal'
    # Add more mappings as needed
}

def map_to_category(objects_str):
    # Check if the value is NaN (float), return 'Other' in such cases
    if pd.isna(objects_str):
        return 'Other'
    
    objects = [obj.strip() for obj in objects_str.split(',')]
    for obj in objects:
        if obj in object_categories:
            return object_categories[obj]
    return 'Other'  # Default category if no match is found



# Assuming your DataFrame is 'filtered_data_11' and 'UTC' is the column with Unix timestamps

# Function to categorize time intervals
def categorize_time(timestamp):
    hour = timestamp.hour
    if 12 <= hour < 15:
        return '12pm-3pm'
    elif 15 <= hour < 18:
        return '3pm-6pm'
    elif 18 <= hour < 21:
        return '6pm-9pm'
    elif 21 <= hour < 24:
        return '9pm-12am'
    elif 0 <= hour < 3:
        return '12am-3am'
    elif 3 <= hour < 6:
        return '3am-6am'
    elif 6 <= hour < 9:
        return '6am-9am'
    elif 9 <= hour < 12:
        return '9am-12pm'
    else:
        return 'Unknown'



def classify_category(text):
    # Use the pre-trained sentiment analysis model
    text_classifier = pipeline("sentiment-analysis")

    result = text_classifier(text)
    
    # Extract the sentiment label
    sentiment_label = result[0]['label']
    
    # Map sentiment labels to categories as needed
    # You can customize this mapping based on your specific use case
    category_mapping = {
        'POSITIVE': 'Positive',
        'NEGATIVE': 'Negative',
        'NEUTRAL': 'Neutral'
    }
    
    return category_mapping.get(sentiment_label, 'Unknown')


# Create a function to normalize 'Upvotes' within each subreddit using min-max scaling
def normalize_upvotes(group):
    min_upvotes = group['Upvotes'].min()
    max_upvotes = group['Upvotes'].max()
    group['Normalized_Upvotes'] = (group['Upvotes'] - min_upvotes) / (max_upvotes - min_upvotes)
    return group

