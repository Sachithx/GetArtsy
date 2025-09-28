# Intern Challenge 1 - GetArtsy
![WGAN_Generated_98](https://user-images.githubusercontent.com/117737754/205827597-5fa6e262-9ce4-4834-8203-a03b3fa74136.png)

# GetArtsy

GetArtsy is a machine learning project focused on generating and analyzing artwork images using Generative Adversarial Networks (GANs). The repository features implementations of several GAN architectures aimed at creating synthetic art images, exploring both unconditional and conditional generation techniques.

## Main Purpose

- **Art Generation:** Create new, high-quality artwork images using advanced GAN models.
- **Artist Classification:** Explore conditional generation by associating generated art with artist labels.
- **Dataset Handling:** Preprocess and manage datasets of artwork images for training GANs.

## Key Features

- **Multiple GAN Architectures:**
  - **DCGAN (Deep Convolutional GAN):** Implementations in both PyTorch (`DCGANs_V2.py`) and TensorFlow/Keras (`DCGANs_V1.py`).
  - **WGAN (Wasserstein GAN):** PyTorch implementation for stable training and improved image quality (`WGAN.py`).
  - **Conditional GANs:** Both Keras (`Conditional_GANs_V1.py`) and PyTorch (`Conditional_GANs_V2.py`) versions to generate images conditioned on artist labels.

- **Dataset Preparation and Preprocessing:**
  - Handles loading, resizing, and label encoding for large image datasets.
  - Support for organizing images by artist and encoding labels for supervised learning.

- **Training Utilities:**
  - Configurable training parameters (epochs, batch size, image dimensions, etc.).
  - Visualization tools for generated images and training progress.

## Technology Stack

- **Languages:** Python
- **Frameworks/Libraries:**
  - PyTorch
  - TensorFlow
  - Keras
  - NumPy
  - OpenCV
  - Matplotlib
  - torchvision

## Example Architectures

- **DCGAN:**  
  Uses convolutional layers for both generator and discriminator, supporting large batch training and high-resolution outputs.

- **WGAN:**  
  Applies Wasserstein loss for improved stability and image generation quality.

- **Conditional GAN:**  
  Conditions the generator and discriminator on artist labels to generate style-specific artwork.

## Dataset Structure

- Images organized in folders by artist.
- Supports encoding of artist names as labels.
- Typical image size for training: 128x128 pixels.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sachithx/GetArtsy.git
   cd GetArtsy
   ```

2. **Prepare your dataset:**
   - Organize artwork images in folders named after artists.
   - Update dataset paths in scripts as needed.

3. **Install required dependencies:**
   ```bash
   pip install torch torchvision tensorflow keras numpy opencv-python matplotlib scikit-learn
   ```

4. **Run a model script:**
   - For unconditional generation:
     ```bash
     python WGAN.py
     ```
   - For conditional generation (Keras):
     ```bash
     python Conditional_GANs_V1.py
     ```
   - For DCGAN (PyTorch):
     ```bash
     python DCGANs_V2.py
     ```

## Repository Details

- **Default Branch:** `main`
- **License:** Not specified
- **Owner:** [Sachithx](https://github.com/Sachithx)
- **Repository:** [Sachithx/GetArtsy](https://github.com/Sachithx/GetArtsy)

## Further Reading

- See each script for detailed architecture and training loop implementations.
- Extend or modify models to fit custom datasets or experiment with new GAN types.

---

_Results above are based on a limited search and may not include all files or features. For a complete view, [browse the code on GitHub](https://github.com/Sachithx/GetArtsy/search?q=)._
