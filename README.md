# Painting_Similarity
# GSoc ArtExtract Task 2: Similarity
# Abstarct
# Approach
**Dataset**
Dataset: https://www.google.com/url?q=https://github.com/NationalGalleryOfArt/opendata&sa=D&source=editors&ust=1743430855089832&usg=AOvVaw2YqeNkDKXA4Gfs70IqK9F0

1️⃣ Optimized Image Loading – The OptimizedImageDataset class ensures efficient image loading, with optional augmentations (flips, rotations, color jittering) for robustness.
2️⃣ Error Handling & Data Integrity – Handles missing or corrupt images gracefully by returning a zero tensor, preventing training crashes.
3️⃣ Automated Image Downloading – The ImageDownloader class fetches painting images from the National Gallery of Art Open Data, with CSV validation and metadata filtering.
4️⃣ Parallelized Image Processing – Uses multi-threaded parallel downloads (16 jobs at once) for fast and scalable image retrieval.
5️⃣ FAISS-Compatible Image Storage – Ensures images are stored in a structured format, enabling efficient similarity searches using FAISS indexing.
**Model**
https://github.com/nithika987/Painting_Similarity/blob/main/models/hybrid_face_similarity.py
Triple-Powered Embeddings:
✔ ArcFace (Identity-Based) – Extracts high-precision facial embeddings, ensuring identity-level similarity for face verification.
✔ DINOv2 (High-Level Semantics) – Captures deep, abstract patterns, making it perfect for matching paintings and artistic styles beyond just pixel similarity.
✔ CLIP (Contextual Understanding) – Bridges visual and textual similarity, enabling text-based image retrieval (e.g., "Find a portrait that looks like Van Gogh's style").

Painting & Face Similarity:
    Painting Matching – Unlike traditional feature extractors, DINOv2 understands texture, brushstrokes, and artistic styles, making it perfect for finding visually similar paintings.
    Face Similarity – ArcFace ensures accurate identity-level matching, while DINOv2 helps find lookalikes even with different lighting, angles, or artistic distortions.

🔹 Superior Image Preprocessing for Maximum Accuracy in Paintings:
     Color & Contrast Enhancement – Uses Adaptive Histogram Equalization (CLAHE) for fine-tuned brightness and contrast correction, ensuring clearer features in paintings & low-light photos.
     Smart Sharpening – Applies Unsharp Masking and High-Pass Filters to refine edges & textures, making face detection and painting details stand out.
     Noise Reduction & Denoising – Removes unwanted grain while preserving fine textures, improving painting comparisons and low-light face matching.

🔹 Robust Face Detection & Extraction:
 Multi-Retry Face Detection 
 Dynamic Bounding Box Optimization 
 
🔹 Optimized for Speed & Efficiency:
 Memory-Efficient Execution 
 Fast Inference with GPU/CPU Flexibility 

 **Image Retrieval: Multi-Model Embedding Search with FAISS**
 https://github.com/nithika987/Painting_Similarity/blob/main/utils/hybrid_retrieval.py
 
1️⃣ Hybrid Feature Extraction – Uses a combination of ArcFace, DINOv2, and CLIP for superior image embeddings.
2️⃣ Efficient Image Indexing – Utilizes FAISS for fast retrieval, ensuring high-speed nearest neighbor searches.
3️⃣ Automated Feature Normalization – Ensures robust and accurate similarity retrieval by normalizing embeddings.
4️⃣ Optimized Query Processing – Dynamically adjusts input images with resizing, normalization, and tensor conversion for inference.
5️⃣ Clear Visual Results – Displays query and similar images with distances, enhancing interpretability.
# Evaluation Metrics
SSIM (Structural Similarity Index Measure): Evaluates perceptual similarity by considering luminance, contrast, and structure, making it ideal for assessing artistic details and textures in paintings.

RMSE (Root Mean Squared Error): Measures pixel-wise differences between images, helping quantify overall visual deviation in paintings.

LPIPS (Learned Perceptual Image Patch Similarity): Uses deep networks to model human visual perception, capturing complex artistic style variations beyond pixel-level comparisons.

Cosine Similarity on Embeddings: Compares feature representations of paintings, ensuring semantic and stylistic similarity detection across different artistic styles.
# Result Analysis
                      Average SSIM Score  Average RMSE Score  Average LPIPS Score
Compressor                                                                 
hybrid_face               0.285237            0.272339             0.597490
hybrid_general            0.285965            0.271676             0.598274

Average Cosine Similarity: 0.8161

The results indicate that the hybrid compression methods achieved an average cosine similarity of 0.8161, suggesting strong feature preservation. However, SSIM (~0.285), RMSE (~0.272), and LPIPS (~0.598) scores reveal moderate perceptual quality, indicating some loss in fine details.

Positives:
Consistent performance across hybrid methods.
Faces detected in many ainintgs, even miniscule faces.

Shortcomings:
Low SSIM suggests structural distortions.
Higher LPIPS indicates perceptual quality degradation.
Faces not detected in all painitngs

![image](https://github.com/user-attachments/assets/138d671e-d2fa-4e48-a146-d124ee412d58)

![image](https://github.com/user-attachments/assets/748688b8-976d-47b7-9295-5c13c94486d0)

![image](https://github.com/user-attachments/assets/50ce1052-3bc7-4cda-949b-9acf51fe0574)

![image](https://github.com/user-attachments/assets/a12586b6-eff9-4c56-82a0-7d9e4946a805)

# Future Scope
Style Transfer & Synthesis – Extend the system to not just find similar paintings but also generate new artworks in a given style using GANs (e.g., StyleGAN, CycleGAN).

Fine-Grained Artist Attribution – Improve artist classification by incorporating hierarchical and contrastive learning techniques to distinguish subtle nuances in painting styles.

Cross-Domain Similarity Search – Expand retrieval to historical manuscripts, sculptures, and photography, creating a unified art similarity framework beyond just paintings.








