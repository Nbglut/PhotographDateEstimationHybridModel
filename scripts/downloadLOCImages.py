import requests
import os
from pathlib import Path
import argparse
import re  # Import the re module for regular expressions

def get_collection_data(collection_name, max_items=50, min_date=None, max_date=None):
    """
    Fetch metadata for items in a LOC collection.

    :param collection_name: The name of the collection (e.g., "civil-war-photographs")
    :param max_items: Maximum number of items to retrieve
    :param min_date: Minimum date for filtering results (format: YYYY-MM-DD)
    :param max_date: Maximum date for filtering results (format: YYYY-MM-DD)
    :return: A list of items from the collection
    """
    BASE_URL = "https://www.loc.gov/collections/"
    QUERY_PARAMS = "?fo=json"
    
    # Construct the URL with optional date filters
    url = f"{BASE_URL}{collection_name}/{QUERY_PARAMS}&c={max_items}"
    print(url)
    if min_date:
        url += f"&dates={min_date}/"
    if max_date:
        url += f"&dates=/{max_date}"

    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data from LOC API: {response.status_code}")
        return None

def sanitize_filename(title):
    """
    Sanitize the title to create a valid filename.

    :param title: The title of the image
    :return: A sanitized filename
    """
    # Remove invalid characters and limit the length
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '_', title)  # Replace invalid characters with underscores
    return sanitized_title[:255]  # Limit filename length to 255 characters

def download_images_from_collection(collection_name, download_dir, max_images=50, min_date=None, max_date=None):
    """
    Download images from a LOC collection.

    :param collection_name: The name of the collection (e.g., "civil-war-photographs")
    :param download_dir: Directory to save downloaded images
    :param max_images: Maximum number of images to download
    :param min_date: Minimum date for filtering results
    :param max_date: Maximum date for filtering results
    """
    data = get_collection_data(collection_name, max_items=max_images, min_date=min_date, max_date=max_date)

    if data:
        items = data.get("results", [])
        for i, item in enumerate(items):
            image_urls = item.get("image_url")
            title = item.get("title", f"image_{i}")

            # Check if image_urls is a list
            if isinstance(image_urls, list) and image_urls:
                image_url = image_urls[0]
            elif isinstance(image_urls, str):
                image_url = image_urls
            else:
                print(f"No valid image URL for item: {title}")
                continue
            
            # Download image content
            try:
                image_data = requests.get(image_url).content
                
                # Create a valid filename from the title
                file_name = sanitize_filename(title) + ".jpg"
                file_path = download_dir / file_name
                
                # Save the image
                with open(file_path, "wb") as img_file:
                    img_file.write(image_data)
                
                print(f"Downloaded: {file_name}")
            except Exception as e:
                print(f"Failed to download image for item: {title}, error: {str(e)}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download images from LOC collections")
    parser.add_argument("collection_name", help="The name of the LOC collection (e.g., 'civil-war-photographs')")
    parser.add_argument("download_dir", type=str, help="Directory to save downloaded images")
    parser.add_argument("--max_images", type=int, default=50, help="Maximum number of images to download")
    parser.add_argument("--min_date", type=str, help="Minimum date for filtering results (format: YYYY-MM-DD)")
    parser.add_argument("--max_date", type=str, help="Maximum date for filtering results (format: YYYY-MM-DD)")
    
    args = parser.parse_args()

    # Create the download directory if it does not exist
    download_dir = Path(args.download_dir)
    download_dir.mkdir(exist_ok=True)

    # Run the downloader
    download_images_from_collection(args.collection_name, download_dir, max_images=args.max_images, min_date=args.min_date, max_date=args.max_date)

if __name__ == "__main__":
    main()
