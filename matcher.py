import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import psycopg2
import os
import warnings

# ==== CONFIGURATION ====
DB_NAME = "aniruddhadhawad"  # replace with your actual DB name
DB_USER = "aniruddhadhawad"
DB_PASSWORD = ""  # add password if required
DB_HOST = "localhost"
DB_PORT = "5432"
DATA_DIR = "/Users/aniruddhadhawad/Library/Application Support/Odoo/"

# ==== Load Pre-trained Model ====
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification head
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# === Utility to safely load and convert images ===
def safe_load_image(image_path):
    """
    Safely load an image and convert it to RGB format.
    Handles palette images with transparency and other edge cases.
    """
    try:
        image = Image.open(image_path)
        
        # Handle palette images with transparency
        if image.mode == 'P':
            if 'transparency' in image.info:
                # Convert palette with transparency to RGBA first
                image = image.convert('RGBA')
            else:
                # Convert palette without transparency to RGB
                image = image.convert('RGB')
        
        # Convert other formats to RGB
        elif image.mode in ('RGBA', 'LA'):
            # Create a white background for images with alpha channel
            background = Image.new('RGB', image.size, (255, 255, 255))
            if image.mode == 'RGBA':
                background.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
            else:  # LA mode
                background.paste(image.convert('RGB'))
            image = background
        
        # Ensure final image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        return None

# === Utility to extract feature vector ===
def extract_features(image_path):
    """Extract feature vector from an image."""
    try:
        image = safe_load_image(image_path)
        if image is None:
            return None
            
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = model(image_tensor).squeeze().numpy()
        return features.reshape(1, -1)
    
    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None

# === Fetch image paths stored in the database ===
def get_image_paths_from_db():
    """Fetch all image attachments from the database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT id, store_fname, name
            FROM ir_attachment
            WHERE mimetype LIKE 'image/%'
            AND store_fname IS NOT NULL
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            {
                "path": os.path.join(DATA_DIR, 'filestore', DB_NAME, store_fname),
                "filename": name,
                "id": id
            }
            for id, store_fname, name in rows
            if store_fname  # Ensure it's not NULL
        ]
    
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []

# === Main matching logic ===
def find_top_matches(reference_image_path, comparison_image_dicts, top_k=3):
    """Find top matching images based on feature similarity."""
    try:
        reference_features = extract_features(reference_image_path)
        if reference_features is None:
            print(f"Could not extract features from reference image: {reference_image_path}")
            return []
            
        similarities = []

        for item in comparison_image_dicts:
            path = item["path"]
            filename = item["filename"]
            id_ = item["id"]
            
            if not os.path.exists(path):
                print(f"File does not exist: {path}")
                continue

            comp_features = extract_features(path)
            if comp_features is None:
                print(f"Could not extract features from: {path}")
                continue
                
            sim = cosine_similarity(reference_features, comp_features)[0][0]
            similarities.append({
                "filename": filename,
                "similarity": float(sim),
                "id": id_,
                "path": path
            })

        # Sort by similarity descending
        top_matches = sorted(similarities, key=lambda x: x["similarity"], reverse=True)[:top_k]
        
        print(f"\nFound {len(top_matches)} matches:")
        for match in top_matches:
            print(f"Match: {match['filename']}, Similarity: {match['similarity']:.4f}, ID: {match['id']}")
            
        return top_matches
    
    except Exception as e:
        print(f"Error in find_top_matches: {str(e)}")
        return []