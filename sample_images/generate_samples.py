"""
Sample Image Generator
======================
Generates academically standard test images covering diverse spectral categories:
1. natural_portrait.png - Smooth gradient portrait
2. landscape.png        - Broad spatial gradients and natural horizon
3. detailed_texture.png - High-frequency oscillatory patterns (fabric / grid)
4. text_document.png    - Sharp binary typography edges
5. geometric_shapes.png - Synthetic geometry with high step contrast
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def generate_all_samples(output_dir: str = "sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    size = (256, 256)

    # 1. Natural Portrait (Smooth Gaussian gradients and concentric features)
    img_portrait = Image.new("L", size, color=240)
    draw_p = ImageDraw.Draw(img_portrait)
    # Head contour
    draw_p.ellipse([50, 40, 206, 220], fill=210, outline=90, width=2)
    # Eyes
    draw_p.ellipse([85, 100, 115, 120], fill=70)
    draw_p.ellipse([141, 100, 171, 120], fill=70)
    # Pupils
    draw_p.ellipse([96, 106, 104, 114], fill=20)
    draw_p.ellipse([152, 106, 160, 114], fill=20)
    # Eyebrows
    draw_p.arc([80, 85, 120, 105], start=200, end=340, fill=40, width=3)
    draw_p.arc([136, 85, 176, 105], start=200, end=340, fill=40, width=3)
    # Nose
    draw_p.line([(128, 115), (122, 150), (134, 150)], fill=100, width=2)
    # Smile
    draw_p.arc([100, 155, 156, 185], start=20, end=160, fill=60, width=3)
    # Add smooth Gaussian blur-like 2D gradient overlay
    arr_p = np.array(img_portrait, dtype=np.float64)
    x = np.linspace(-2, 2, size[0])
    y = np.linspace(-2, 2, size[1])
    xx, yy = np.meshgrid(x, y)
    vignette = np.exp(-(xx**2 + yy**2) / 3.0)
    arr_p = arr_p * (0.7 + 0.3 * vignette)
    Image.fromarray(np.clip(arr_p, 0, 255).astype(np.uint8)).save(
        os.path.join(output_dir, "natural_portrait.png")
    )

    # 2. Landscape (Sky gradient + mountains + foreground)
    arr_land = np.zeros(size, dtype=np.uint8)
    for row in range(size[0]):
        # Sky smooth gradient from light to medium
        arr_land[row, :] = int(180 - row * 0.4)
    img_land = Image.fromarray(arr_land)
    draw_l = ImageDraw.Draw(img_land)
    # Mountain peaks
    mountains = [(0, 140), (40, 100), (80, 130), (130, 80), (180, 135), (220, 110), (256, 145), (256, 256), (0, 256)]
    draw_l.polygon(mountains, fill=90, outline=40)
    # Foreground hill
    hills = [(0, 180), (70, 165), (150, 190), (256, 170), (256, 256), (0, 256)]
    draw_l.polygon(hills, fill=50, outline=20)
    # Sun
    draw_l.ellipse([180, 30, 225, 75], fill=255)
    img_land.save(os.path.join(output_dir, "landscape.png"))

    # 3. Detailed Texture (High-frequency periodic weave and noise)
    x = np.arange(size[0])
    y = np.arange(size[1])
    xx, yy = np.meshgrid(x, y)
    freq1 = 0.25
    freq2 = 0.5
    texture = (
        128.0
        + 60.0 * np.sin(freq1 * xx) * np.cos(freq1 * yy)
        + 40.0 * np.cos(freq2 * xx + freq2 * yy)
        + 20.0 * np.sin(0.1 * xx * yy / 10.0)
    )
    # Add subtle checkerboard high frequency
    checker = ((xx // 4) % 2 == (yy // 4) % 2) * 25.0
    texture += checker
    Image.fromarray(np.clip(texture, 0, 255).astype(np.uint8)).save(
        os.path.join(output_dir, "detailed_texture.png")
    )

    # 4. Text Document (High-contrast typography and sharp edges)
    img_text = Image.new("L", size, color=255)
    draw_t = ImageDraw.Draw(img_text)
    # Title
    draw_t.text((20, 20), "MCA APPLIED MATHEMATICS", fill=0)
    draw_t.line([(20, 35), (236, 35)], fill=0, width=2)
    # Simulated paragraphs
    lines = [
        "Linear Algebra & Matrix Compression",
        "Singular Value Decomposition: A = U Sigma V^T",
        "Rank-k Low-Rank Truncation",
        "Eckart-Young-Mirsky Theorem (1936)",
        "Discrete Cosine Transform (DCT-II)",
        "8x8 Block Quantization & Entropy",
        "Statistical PSNR and MSE Verification",
        "Theoretical Representation vs File Size",
        "Energy Compaction into Low Frequencies",
        "Block Artifacts vs Directional Smear"
    ]
    for idx, text in enumerate(lines):
        y_pos = 50 + idx * 19
        draw_t.text((20, y_pos), text, fill=0)
        # Small decorative box
        draw_t.rectangle([220, y_pos + 2, 230, y_pos + 12], outline=0, width=1)
    img_text.save(os.path.join(output_dir, "text_document.png"))

    # 5. Geometric Shapes (Circles, polygons, sharp step functions)
    img_geo = Image.new("L", size, color=230)
    draw_g = ImageDraw.Draw(img_geo)
    draw_g.rectangle([20, 20, 90, 90], fill=40, outline=0, width=2)
    draw_g.ellipse([140, 20, 230, 110], fill=120, outline=0, width=2)
    draw_g.polygon([(128, 140), (70, 230), (186, 230)], fill=80, outline=0)
    draw_g.line([(10, 10), (246, 246)], fill=0, width=3)
    draw_g.line([(10, 246), (246, 10)], fill=160, width=3)
    img_geo.save(os.path.join(output_dir, "geometric_shapes.png"))

    print(f"Successfully generated 5 academic sample images in '{output_dir}'.")


if __name__ == "__main__":
    generate_all_samples()
