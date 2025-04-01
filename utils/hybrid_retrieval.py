import os
import torch
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import faiss
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader

class HybridImageRetrieval:
    """
    Improved Image Retrieval System using a Hybrid Model (ArcFace + DINOv2 + CLIP).
    
    Args:
        model_path (str): Path to the hybrid model.
        img_path (str): Path to the directory containing images.
        k (int): Number of similar images to retrieve (default: 5).
        device (str): Device to run the model (default: 'cpu').

    Attributes:
        model (HybridFaceSimilarity): Hybrid feature extraction model.
        img_path (str): Path to the directory containing images.
        k (int): Number of similar images to retrieve.
        device (str): Device for running the model.
        merged_df (pd.DataFrame): DataFrame containing image metadata.
        features (np.ndarray): Extracted features from all images.
        image_paths (list): List of image paths.
        faiss_index (faiss.IndexFlatIP): FAISS index for fast retrieval.
    """

    def __init__(self, model_path, img_path, k=5, device='cpu'):
        
        self.model = HybridFaceSimilarity(device=device)
        self.device = device
        self.img_path = img_path
        self.k = k
        self.merged_df = pd.read_csv("/kaggle/working/data/annotations/merged_filtered.csv")

        # Extract and index features
        self.features, self.image_paths = self.extract_features()
        self.build_faiss_index()

    def extract_features(self):
        """
        Extracts and indexes features from all images using the hybrid model.
        """
        image_dataset = OptimizedImageDataset(self.merged_df, self.img_path, augment=False)
        data_loader = DataLoader(image_dataset, batch_size=32, shuffle=False)
    
        features, image_paths = [], []
    
        print("Extracting Features from Images...")
        for image_tensors in tqdm(data_loader, total=len(data_loader)):
            image_tensors = image_tensors.to(self.device)

            features_list = []
            for img in image_tensors:
                extracted = self.model.hybrid_embedding(img)  # Dictionary output
                embedding = extracted["embedding"]  # Extract tensor before further processing
                features_list.append(embedding)
    
            # Stack all features into a single tensor
            extracted_features = torch.stack(features_list).to(self.device)
    
            # Convert to NumPy
            extracted_features = extracted_features.view(extracted_features.size(0), -1).cpu().numpy()
    
            features.extend(extracted_features)
            image_paths.extend(image_dataset.dataFrame["objectid"].tolist())
    
        return np.array(features), image_paths

    def build_faiss_index(self):
        """
        Builds a FAISS index for fast retrieval of nearest neighbors.
        """
        print("Building FAISS Index...")
    
        # Always use 1792 for face embeddings
        faiss_index_dimension = 1792
    
        # Create the FAISS index with 1792 dimensions
        self.faiss_index = faiss.IndexFlatIP(faiss_index_dimension)
    
        # Normalize features before adding to FAISS
        self.features = self.features / np.linalg.norm(self.features, axis=1, keepdims=True)
    
        # Add normalized features to the FAISS index
        self.faiss_index.add(self.features.astype(np.float32))
    
        print("FAISS Index built successfully with dimension 1792!")


    def retrieve_similar_images(self, query_image_path, metric="cosine"):
        """
        Retrieves similar images to a query using the hybrid model.
        """
        # Pre-process query image
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = Image.open(query_image_path).convert("RGB")
        query_image_tensor = self.transform(image).unsqueeze(0).to(self.device)
    
        # Extract query features
        query_result = self.model.hybrid_embedding(query_image_tensor)
        query_features = query_result["embedding"].cpu().numpy()
    
        # Debugging: Check shape before reshape
        #print("Query Features Shape (Before Reshape):", query_features.shape)

        # Ensure query_features is (1, feature_dim)
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        elif query_features.ndim == 0:
            raise ValueError("Extracted query features are empty!")
    
        #print("Query Features Shape (After Reshape):", query_features.shape)
    
        # Normalize query features before FAISS search
        query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
        
        # Ensure FAISS dimension matches
        if query_features.shape[1] != self.faiss_index.d:
            raise ValueError(f"FAISS dimension mismatch! Expected {self.faiss_index.d}, got {query_features.shape[1]}.")
    
        # Search for nearest neighbors using FAISS
        D, I = self.faiss_index.search(query_features.astype(np.float32), self.k + 1)
    
        # Retrieve similar images (skip self-match)
        similar_images = [
            (self.image_paths[idx], D[0][j]) for j, idx in enumerate(I[0]) if idx < len(self.image_paths)
        ][1:]  
    
        return similar_images
     
    def visualize_results(self, similar_images, query_image_path):
        """
        Visualizes query image alongside its retrieved similar images.

        Args:
            similar_images (list): List of tuples containing image paths and distances.
            query_image_path (str): Path to the query image.
        """
        query_image = Image.open(query_image_path)

        fig, axes = plt.subplots(1, len(similar_images) + 1, figsize=(15, 5))
        axes[0].imshow(query_image)
        axes[0].set_title("Query Image")
        axes[0].axis("off")

        for i, (file_name, distance) in enumerate(similar_images, start=1):
            image_path = os.path.join(self.img_path, str(file_name) + ".jpg")
            image = Image.open(image_path)
            axes[i].imshow(image)
            axes[i].set_title(f"Similar Image {i}\nDist: {distance:.4f}")
            axes[i].axis("off")

        plt.tight_layout()
        plt.show()
