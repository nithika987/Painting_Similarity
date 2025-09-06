

# 🖌️ Painting_Similarity

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📝 Abstract
This project implements a **hybrid painting similarity system** using **ArcFace**, **DINOv2**, and **CLIP embeddings** for feature extraction.  
It leverages **FAISS indexing** for fast retrieval of visually and stylistically similar artworks.  

**Applications:**  
- 🎨 Art analysis & recommendation  
- 🖼 Historical research  
- 🔍 Efficient painting matching based on learned representations  

---

## 📦 Dataset
- **Source:** [National Gallery of Art Open Data](https://github.com/NationalGalleryOfArt/opendata)  

**Preprocessing & Loading:**
1. 🖼 **Optimized Image Loading:** `OptimizedImageDataset` with optional augmentations (flip, rotation, color jitter)  
2. ⚠️ **Error Handling:** Handles missing/corrupt images with zero tensors  
3. 🌐 **Automated Downloading:** `ImageDownloader` fetches paintings with CSV validation  
4. ⚡ **Parallelized Processing:** Multi-threaded downloads (16 jobs at once)  
5. 💾 **FAISS-Compatible Storage:** Structured for efficient similarity search  

---

## 🏗 Model Architecture
- **Implementation:** [hybrid_face_similarity.py](https://github.com/nithika987/Painting_Similarity/blob/main/models/hybrid_face_similarity.py)

**Triple-Powered Embeddings:**  
- 🔹 **ArcFace (Identity-Based):** High-precision facial embeddings for identity similarity  
- 🔹 **DINOv2 (High-Level Semantics):** Captures deep, abstract patterns, making it perfect for matching paintings and artistic styles beyond just pixel similarity.  
- 🔹 **CLIP (Contextual Understanding):** Bridges visual & textual similarity for text-based retrieval (e.g., "Find a portrait that looks like Van Gogh's style").

**Painting & Face Similarity:**  
- 🖌 **Painting Matching:** DINOv2 understands texture, brushstrokes, and artistic style (for visual similarity). 
- 🧑 **Face Similarity:** ArcFace ensures identity-level matching; DINOv2 finds lookalikes even with different lighting, angles, or artistic distortions.

**Advanced Image Preprocessing:**  
- 🎨 **Color & Contrast Enhancement:** CLAHE for brightness/contrast correction  
- ✨ **Smart Sharpening:** Unsharp Mask & High-Pass filters for edges & textures  
- 🧹 **Noise Reduction:** Preserves fine textures while removing unwanted grain  

**Robust Face Detection:**  
- 🔁 Multi-retry face detection  
- 📦 Dynamic bounding box optimization  

**Optimized for Speed:**  
- ⚡ Memory-efficient execution  
- 🖥 GPU/CPU flexible inference  

**Image Retrieval with FAISS:**  
- Hybrid feature extraction: ArcFace + DINOv2 + CLIP for superior image embeddings
- Efficient image indexing with FAISS;  ensuring high-speed nearest neighbor searches
- Feature normalization for robust similarity  
- Optimized query processing (resize, normalize, tensor conversion)  
- Clear visual results with distances  

[Code: hybrid_retrieval.py](https://github.com/nithika987/Painting_Similarity/blob/main/utils/hybrid_retrieval.py)  

---

## 📊 Evaluation Metrics
- **SSIM:** Measures perceptual similarity (luminance, contrast, structure)  
- **RMSE:** Pixel-wise difference for visual deviation  
- **LPIPS:** Learned perceptual similarity capturing artistic style variations  
- **Cosine Similarity:** Compares embeddings for semantic/style similarity  

**Results:**

| Compressor         | Avg SSIM | Avg RMSE | Avg LPIPS |
|------------------|-----------|----------|-----------|
| **hybrid_face**    | 0.285237  | 0.272339 | 0.597490  |
| **hybrid_general** | 0.285965  | 0.271676 | 0.598274  |

**Average Cosine Similarity:** 0.8161  

**Observations:**  
- ✅ Consistent performance across hybrid methods  
- ✅ Faces detected even in very small paintings  
- ⚠️ Low SSIM indicates structural distortions  
- ⚠️ Higher LPIPS shows perceptual quality degradation  
- ⚠️ Some faces not detected  

---

## 🖼 Test Samples

**Test 1:**  

![image](https://github.com/user-attachments/assets/138d671e-d2fa-4e48-a146-d124ee412d58)  

**Test 2:**  

![image](https://github.com/user-attachments/assets/748688b8-976d-47b7-9295-5c13c94486d0)  

**Test 3:** 

![image](https://github.com/user-attachments/assets/50ce1052-3bc7-4cda-949b-9acf51fe0574)  

**Face Detection Example:**  

![image](https://github.com/user-attachments/assets/a12586b6-eff9-4c56-82a0-7d9e4946a805)  

---

## 🌟 Future Scope
- 🎨 **Style Transfer & Synthesis:** Generate new artworks in a given style (GANs: StyleGAN, CycleGAN)  
- 🖌 **Fine-Grained Artist Attribution:** Hierarchical & contrastive learning for subtle style distinctions  
- 🌍 **Cross-Domain Similarity Search:** Expand to manuscripts, sculptures, and photography  

---

## 🏷 Tags
`#DeepLearning` `#ComputerVision` `#ArtSimilarity` `#ArcFace` `#DINOv2` `#CLIP` `#FAISS` `#Python` `#PyTorch`  

---

## 📄 License
This project is licensed under the **MIT License** – see the [LICENSE](https://github.com/nithika987/Painting_Similarity/blob/main/LICENSE) file for details.









