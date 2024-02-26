import os
import requests
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from webcolors import hex_to_rgb, CSS3_HEX_TO_NAMES
import shutil
from collections import Counter

def fetch_image_from_url(url):
    """Fetch an image from a URL and return it as a NumPy array."""
    response = requests.get(url)
    if response.status_code == 200:
        return cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)

def color_difference(color1, color2):
    """Calculate the Euclidean distance between two color vectors."""
    return sum((a - b) ** 2 for a, b in zip(color1, color2))

def find_closest_color_name(rgb_value):
    """Find the closest CSS3 color name to an RGB value."""
    closest_color = min(CSS3_HEX_TO_NAMES, key=lambda name: color_difference(rgb_value, hex_to_rgb(name)))
    return CSS3_HEX_TO_NAMES[closest_color]

def extract_top_colors(img, num_colors=5):
    """Extract the top N dominant colors from an image and match them to the closest CSS3 color names."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img_array)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    color_names = [find_closest_color_name(tuple(color)) for color in dominant_colors]
    return dominant_colors, color_names

def process_images(df, output_csv):
    """Process images in a DataFrame and save the results to a CSV file."""
    dominant_colors_list = []
    color_names_list = []

    for image_url in df['Image URL']:
        try:
            img = fetch_image_from_url(image_url)
            if img is not None and img.size >= 100:
                dominant_colors, color_names = extract_top_colors(img)
                dominant_colors_list.append(str(dominant_colors))
                color_names_list.append(str(color_names))
            else:
                dominant_colors_list.append(None)
                color_names_list.append(None)
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
            dominant_colors_list.append(None)
            color_names_list.append(None)

    df['Dominant Colors'] = dominant_colors_list
    df['Color Names'] = color_names_list
    df.to_csv(output_csv, index=False)

def copy_files_with_color_check(csv_file_path, source_folder, destination_folder):
    """Copy files based on color checks from a CSV file."""
    df = pd.read_csv(csv_file_path)
    os.makedirs(destination_folder, exist_ok=True)

    for _, row in df.iterrows():
        filename = row['Filename']
        color_names = row.get('Color Names', '')
        if pd.notna(color_names) and color_names.strip():
            source_filepath = os.path.join(source_folder, filename)
            destination_filepath = os.path.join(destination_folder, filename)
            if os.path.exists(source_filepath):
                shutil.copy(source_filepath, destination_filepath)
                print(f"File '{filename}' copied successfully.")
            else:
                print(f"File '{filename}' not found in the source folder.")
        else:
            print(f"Skipping file '{filename}' due to empty Color Names.")


def get_image_formats_distribution(folder_path):
    image_formats = []
    
    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            # Get the file extension (image format)
            _, file_extension = os.path.splitext(file_name)
            image_formats.append(file_extension.lower())

    # Count the occurrences of each image format
    image_formats_distribution = Counter(image_formats)
    
    return image_formats_distribution

# Function to copy images from one folder to another
def copy_images(source_folder, destination_folder):
    for folder_name in os.listdir(source_folder):
        folder_path = os.path.join(source_folder, folder_name)
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    source_file_path = os.path.join(folder_path, file_name)
                    destination_file_path = os.path.join(destination_folder, file_name)
                    shutil.copy(source_file_path, destination_file_path)
   



if __name__ == "__main__":
    main()
