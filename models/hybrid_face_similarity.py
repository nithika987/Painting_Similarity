import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from transformers import CLIPProcessor, CLIPModel
from insightface.app import FaceAnalysis
from torchvision.transforms.functional import to_pil_image
import gc
import numpy as np
import cv2

class HybridFaceSimilarity:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Limit GPU memory usage to 70% to leave room for spikes
        if device == 'cuda':
            torch.cuda.set_per_process_memory_fraction(0.7, device=0)

        # MTCNN for face detection with relaxed thresholds
        self.face_detector = MTCNN(keep_all=True, device=device, min_face_size=3, thresholds=[0.5, 0.5, 0.6], post_process=True)

        # ArcFace for identity-based facial features
        self.arcface = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider']
        )
        self.arcface.prepare(ctx_id=0 if device == 'cuda' else -1)

        # Load DINOv2 with vit_base and move to CPU to save GPU memory
        from dinov2.models.vision_transformer import vit_base
        self.dino = vit_base().to('cpu').eval().half()

        # CLIP for text-image similarity (half precision to reduce memory)
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval().half()
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def clear_memory(self):
        """ Clear CUDA memory and free unused objects """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def adaptive_resize(self, image, max_size=512):
        """ Dynamically resize large images to reduce memory load """
        w, h = image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            image = image.resize(new_size)
        return image

    def enhance_contrast(self, image):
        """ Improve contrast and sharpness for better face detection in blurry images """
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
        # Adaptive Histogram Equalization (CLAHE) to enhance local contrast
        lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
        # Unsharp Masking for sharpening
        gaussian_blur = cv2.GaussianBlur(img_cv, (0, 0), 2.0)
        img_cv = cv2.addWeighted(img_cv, 1.5, gaussian_blur, -0.5, 0)
    
        # Bilateral Filtering to remove noise while keeping edges
        img_cv = cv2.bilateralFilter(img_cv, 9, 75, 75)
    
        # Convert back to PIL format
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    def apply_denoising(self, image):
        """ Apply noise reduction before detecting faces """
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
        # Non-Local Means Denoising (best for reducing grainy noise)
        img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)
    
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    def sharpen_image(self, image):
        """ Apply sharpening to improve face edges """
        img_cv = np.array(image)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    
        # High-pass filter for sharpening
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img_cv = cv2.filter2D(img_cv, -1, kernel)
    
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


    def extract_faces(self, image, max_retries=3):
        """ Detect and crop faces with automatic contrast enhancement retries """
        if isinstance(image, torch.Tensor):
            image = to_pil_image(image.squeeze(0))
        else:
            image = image.convert("RGB")
    
        image = self.adaptive_resize(image)  # Resize large images
    
        for attempt in range(max_retries):
            if attempt > 0:
                #print(f"No faces detected. Retrying with enhanced preprocessing (Attempt {attempt})...")
    
                if attempt == 1:
                    image = self.enhance_contrast(image)  # Apply contrast boost
                elif attempt == 2:
                    image = self.sharpen_image(image)  # Apply sharpening
                elif attempt == 3:
                    image = self.apply_denoising(image)  # Apply noise reduction
    
            # Face detection with MTCNN
            boxes, probs = self.face_detector.detect(image)
    
            if boxes is not None and len(boxes) > 0:
                #print(f"{len(boxes)} face(s) detected.")
                break  # Faces detected, stop retrying
    
        if boxes is None or len(boxes) == 0:
            #print("No faces detected even after multiple enhancements.")
            return []
    
        faces = []
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
    
            # Compute face size and apply dynamic margin
            face_size = max(x_max - x_min, y_max - y_min)
            margin = int(0.15 * face_size)
    
            # Adjust bounding box
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(image.width, x_max + margin)
            y_max = min(image.height, y_max + margin)
    
            # Convert to square bounding box
            side = max(x_max - x_min, y_max - y_min)
            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            x_min, x_max = cx - side // 2, cx + side // 2
            y_min, y_max = cy - side // 2, cy + side // 2
    
            # Ensure bounding box is within image limits
            x_min, x_max = max(0, x_min), min(image.width, x_max)
            y_min, y_max = max(0, y_min), min(image.height, y_max)
    
            # Crop and resize the face
            face = image.crop((x_min, y_min, x_max, y_max)).resize((224, 224))
            faces.append(face)
    
        #print(f"{len(faces)} face(s) detected after enhancement.")
        return faces


    def extract_arcface(self, image):
        """ Extract ArcFace embeddings in batch to save memory """
        faces = self.extract_faces(image)
        features = []
        batch_faces = []

        for face in faces:
            face_np = np.array(face)
            aligned_face = self.arcface.get(face_np)
            if aligned_face:
                batch_faces.append(aligned_face[0]['embedding'])

            # Process in batch to avoid memory spike
            if len(batch_faces) >= 8:
                batch_faces_tensor = torch.tensor(batch_faces, device=self.device)
                features.append(batch_faces_tensor)
                batch_faces = []

        if batch_faces:
            batch_faces_tensor = torch.tensor(batch_faces, device=self.device)
            features.append(batch_faces_tensor)

        if features:
            features_tensor = torch.cat(features, dim=0)
            del batch_faces, batch_faces_tensor
            self.clear_memory()
            return features_tensor
        else:
            return torch.zeros((1, 512), device=self.device)  # Return empty tensor if no face

    def extract_dino(self, image):
        """ Extract DINOv2 features (GPU inference if available) """
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = to_pil_image(image)

        image_tensor = self.transform(image).unsqueeze(0).to(self.device).half()
        self.dino = self.dino.to(self.device).half()

        with torch.no_grad():
            features = self.dino(image_tensor)

        del image_tensor
        self.clear_memory()

        return features.flatten()

    def extract_clip(self, image, text_prompt=None):
        """ Extract CLIP embeddings (image and optional text similarity) """
        if isinstance(image, torch.Tensor):
            image = (image + 1) / 2
            image = torch.clamp(image, 0, 1)

        elif isinstance(image, np.ndarray):
            image = np.clip((image + 1) / 2, 0, 1)

        #image_input = self.clip_processor(images=image, return_tensors="pt")
        image_input = self.clip_processor(images=image, return_tensors="pt", do_rescale=False)

        for key in image_input:
            image_input[key] = image_input[key].to(self.device).half()

        with torch.no_grad():
            image_features = self.clip.get_image_features(**image_input).squeeze()

        return image_features

    def hybrid_embedding(self, image, text_prompt=None):
        """ Generate a combined embedding from ArcFace, DINOv2, and CLIP """
        with torch.no_grad():
            arcface_embedding = self.extract_arcface(image)
            arcface_mean = arcface_embedding.mean(dim=0) if len(arcface_embedding.shape) > 1 else arcface_embedding

            dino_embedding = self.extract_dino(image)
            clip_embedding = self.extract_clip(image, text_prompt)

            combined_embedding = torch.cat([
                arcface_mean.to(self.device),
                dino_embedding.to(self.device),
                clip_embedding
            ], dim=0)

            self.clear_memory()

        return {"embedding": combined_embedding, "clip_similarity": None if text_prompt is None else clip_embedding}

