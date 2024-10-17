import os
import requests
from tqdm import tqdm
from datetime import datetime

# Wikimedia Commons API URL
API_URL = "https://commons.wikimedia.org/w/api.php"

# Function to create decade-specific folders
def create_folders():
    for decade in range(1880, 2010, 10):
        folder_name = f"{decade}s"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

# Function to get month name from month number
def get_month_name(month):
    return datetime(1900, month, 1).strftime('%B')  # Get full month name

# Function to fetch image URLs from Wikimedia Commons subcategories
def fetch_images_by_decade(decade, max_images=500):
    images = {}
    continue_token = None
    total_images_fetched = 0

    # Fetch images for each year in the decade
    for year in range(decade, decade + 10):
        for month in range(1, 13):
            month_name = get_month_name(month)
            category_title = f"Category:{month_name}_{year}_photographs"

            print(f"Fetching images for {category_title}...")

            while total_images_fetched < max_images:
                params = {
                    'action': 'query',
                    'format': 'json',
                    'prop': 'imageinfo',
                    'iiprop': 'url|extmetadata',
                    'generator': 'categorymembers',
                    'gcmtitle': category_title,
                    'gcmnamespace': '6',  # Files namespace
                    'gcmtype': 'file',
                    'gcmlimit': '50',  # Fetch 50 images per request
                }

                if continue_token:
                    params['gcmcontinue'] = continue_token  # Use pagination token if available

                try:
                    response = requests.get(API_URL, params=params)
                    response.raise_for_status()  # Ensure the request is successful
                    data = response.json()

                    if 'query' in data and 'pages' in data['query']:
                        new_images = data['query']['pages']
                        images.update(new_images)  # Add new images to the collection
                        total_images_fetched += len(new_images)  # Track total images

                        if total_images_fetched >= max_images:
                            break  # Stop if we have fetched enough images

                    # Continue pagination if available
                    if 'continue' in data:
                        continue_token = data['continue'].get('gcmcontinue')
                    else:
                        break  # Exit loop if there's no more data to fetch

                except requests.exceptions.RequestException as e:
                    print(f"Error fetching images for {category_title}: {e}")
                    break

    print(f"Fetched {total_images_fetched} images for the {decade}s")
    return images

# Function to sanitize file names by removing spaces and invalid characters
def sanitize_file_name(file_name):
    return file_name.replace(" ", "_")

# Function to download images into their respective decade folder
def download_images(images, decade):
    downloaded_count = 0
    folder_name = f"{decade}s"  # Decade folder

    for image in tqdm(images.values(), desc=f"Downloading images for {decade}s"):
        if downloaded_count >= 500:
            break  # Stop once we reach the limit

        image_info = image.get('imageinfo', [])[0]

        # Fetch the best available image URL (original if possible)
        image_url = image_info['url']
        image_title = image['title'].replace('File:', '')

        # Sanitize the file name by removing spaces
        sanitized_image_title = sanitize_file_name(image_title)

        # Add proper extension if not provided (default to jpg)
        if not sanitized_image_title.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            sanitized_image_title += ".jpg"

        # Debugging: Print the image URL
        print(f"Attempting to download image from URL: {image_url}")

        # Download and save the image directly
        file_path = os.path.join(folder_name, sanitized_image_title)
        try:
            download_image(image_url, file_path)
            downloaded_count += 1
        except Exception as e:
            print(f"Failed to download {sanitized_image_title}: {e}")

# Function to download and save an image from a URL
def download_image(image_url, file_path):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Request the image URL and follow redirects
        response = requests.get(image_url, stream=True, headers=headers, allow_redirects=True)
        content_type = response.headers.get('Content-Type')

        # Ensure the content type is an image
        if 'image' not in content_type:
            raise ValueError(f"URL does not point to an image. Content-Type: {content_type}")

        # Write the image content to file
        with open(file_path, 'wb') as out_file:
            for chunk in response.iter_content(1024):
                out_file.write(chunk)

        print(f"Downloaded {file_path}")

    except Exception as e:
        print(f"Error downloading image from {image_url}: {e}")

# Main function to download images for each decade
def main():
    create_folders()

    for decade in range(1880, 2010, 10):
        print(f"Gathering images for the {decade}s...")
        images = fetch_images_by_decade(decade)

        if images:
            download_images(images, decade)
        else:
            print(f"No images found for the {decade}s.")

if __name__ == "__main__":
    main()

