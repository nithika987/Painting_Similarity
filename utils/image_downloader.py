import os
from tqdm import tqdm
import pandas as pd
import urllib.request
from math import floor
from joblib import Parallel, delayed
from PIL import Image


class ImageDownloader:
    """
    Enhanced utility class for downloading and managing painting images from the National Gallery of Art Open Data.

    Attributes:
        loader_root (str): Root directory where images and metadata are stored.
        csv_remote_path (str): Remote path for CSV files (NationalGalleryOfArt/opendata).

    Methods:
        ensure_exists(self, path, image=False): Ensure that the specified directory exists.
        get_base_dir(self): Get the base directory and create necessary subdirectories.
        thumbnail_to_local(self, base_path, object_id): Generate the local path for storing a thumbnail image.
        get_file(self, remote_url, out, timeout_seconds=10): Download a file from a remote URL.
        validate_image(self, img_path): Validate the integrity of downloaded images.
        check_csv_exists(self, csv_name, base_dir=None): Check if a CSV file exists locally, and download if not.
        download_painting(self, base_dir=None, percent=100, element_filter='painted surface'): Download paintings.
        merge_and_filter(self, objects_df, images_df): Merge and filter DataFrames for relevant data.
    """

    def __init__(self, loader_root):
        self.loader_root = loader_root
        self.csv_remote_path = 'https://raw.githubusercontent.com/NationalGalleryOfArt/opendata/main/data/'

    def ensure_exists(self, path, image=False):
        """Ensure that the specified directory exists."""
        if not os.path.exists(path):
            os.makedirs(path)
        elif os.listdir(path) and image:
            raise OSError(f"The folder '{path}' is not empty. Clear it before running the script again.")

    def get_base_dir(self):
        """Get the base directory and create necessary subdirectories."""
        self.ensure_exists(self.loader_root)
        self.ensure_exists(f"{self.loader_root}/annotations")
        self.ensure_exists(f"{self.loader_root}/images", image=True)
        return self.loader_root

    def thumbnail_to_local(self, base_path, object_id):
        """Generate the local path for storing a thumbnail image."""
        image_path = f"{base_path}/images"
        return f"{image_path}/{object_id}.jpg"

    def get_file(self, remote_url, out, timeout_seconds=10):
        """Download a file from a remote URL."""
        try:
            with urllib.request.urlopen(remote_url, timeout=timeout_seconds) as response:
                with open(out, "wb") as out_file:
                    out_file.write(response.read())
        except Exception as e:
            print(f"Error downloading {remote_url}: {e}")

    def validate_image(self, img_path):
        """Validate downloaded images to ensure integrity."""
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify image integrity
            return True
        except Exception:
            os.remove(img_path)  # Remove corrupt images
            return False

    def check_csv_exists(self, csv_name, base_dir=None):
        """Check if a CSV file exists locally and download it if not."""
        base_dir = base_dir or self.get_base_dir()
        csv_path = f"{base_dir}/annotations/{csv_name}.csv"

        if not os.path.exists(csv_path):
            print(f"Downloading {csv_name}.csv...")
            self.get_file(f"{self.csv_remote_path}/{csv_name}.csv", csv_path, timeout_seconds=100)
            print(f"{csv_name}.csv downloaded successfully!")
        return csv_path

    def download_painting(self, base_dir=None, percent=100, element_filter='painted surface'):
        """
        Download painting images and associated metadata from the National Gallery of Art Open Data.

        Args:
            base_dir (str, optional): Base directory path.
            percent (int, optional): Percentage of data to download.
            element_filter (str, optional): Filter based on element (e.g., 'painted surface').
        """
        print("Initializing download...")
        base_dir = base_dir or self.get_base_dir()

        # Load CSV files
        objects_csv = self.check_csv_exists('objects_dimensions', base_dir)
        images_csv = self.check_csv_exists('published_images', base_dir)
        objects_df = pd.read_csv(objects_csv)
        images_df = pd.read_csv(images_csv)

        # Merge and filter data
        painted_df = self.merge_and_filter(objects_df, images_df, element_filter)
        samples = floor(painted_df.shape[0] * (percent / 100))
        painted_df = painted_df.head(samples)

        def download_and_validate(object_id, thumb_url):
            out_path = self.thumbnail_to_local(base_dir, object_id)
            if os.path.exists(out_path):
                if self.validate_image(out_path):
                    return  # Skip if image is valid
                else:
                    print(f"Corrupted image found. Re-downloading {object_id}.jpg")

            try:
                self.get_file(thumb_url, out_path)
                if not self.validate_image(out_path):
                    raise ValueError("Downloaded image is corrupt.")
            except Exception as e:
                print(f"Failed to download {thumb_url}: {e}")

        # Download images in parallel
        print(f"Found {painted_df['objectid'].nunique()} valid images to download.")
        Parallel(n_jobs=16)(
            delayed(download_and_validate)(object_id, thumb_url)
            for object_id, thumb_url in tqdm(painted_df[['objectid', 'iiifthumburl']].values)
            
        )

        # Handle missing or failed downloads
        existing_files = os.listdir(os.path.join(self.loader_root, 'images'))
        existing_objectids = {int(filename.split('.')[0]) for filename in existing_files}
        missing_objectids = set(painted_df['objectid']) - existing_objectids
        painted_df = painted_df[~painted_df['objectid'].isin(missing_objectids)]

        # Save filtered DataFrame to CSV
        output_path = f"{self.loader_root}/annotations/merged_filtered.csv"
        painted_df.to_csv(output_path, index=False)
        print(f"{len(missing_objectids)} images failed to download.")
        print(f"{len(painted_df['objectid'])} valid images downloaded successfully.")

    def merge_and_filter(self, objects_df, images_df, element_filter):
        """
        Merge and filter DataFrames to get relevant painting data.

        Args:
            objects_df (DataFrame): DataFrame containing object information.
            images_df (DataFrame): DataFrame containing image URLs.
            element_filter (str): Element type to filter paintings (default: 'painted surface').

        Returns:
            DataFrame: Filtered DataFrame with relevant painting data.
        """
        painted_df = pd.merge(
            objects_df[['objectid', 'element']],
            images_df[['depictstmsobjectid', 'iiifthumburl']],
            left_on='objectid', right_on='depictstmsobjectid',
            how='inner'
        ).query(f"element == '{element_filter}'")

        painted_df = painted_df.drop_duplicates().drop('depictstmsobjectid', axis=1)
        return painted_df
