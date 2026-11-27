"""
Mathematical Image Compression Using SVD and DCT
=================================================
Interactive Streamlit Application for Academic Demonstration,
Laboratory Evaluation, and MCA Project Viva Defense.
"""

import os
import sys
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from processing.image_loader import load_image, pad_to_block_size, crop_to_original_size
from processing.preprocessing import to_float_matrix, to_uint8_image
from algorithms.svd_engine import (
    compute_svd_thin,
    reconstruct_svd_rank_k,
    calculate_energy_retention,
    svd_storage_analysis,
    compress_svd_image,
    verify_svd_on_small_matrix
)
from algorithms.dct_engine import (
    JPEG_LUMINANCE_QUANT_TABLE,
    DCT_MATRIX_8X8,
    scale_quantization_matrix,
    compress_dct_image,
    verify_dct_on_small_block
)
from evaluation.metrics import compute_all_metrics
from evaluation.storage_analysis import analyze_svd_storage, analyze_dct_storage
from evaluation.benchmarking import benchmark_svd_ranks, benchmark_dct_qualities
from visualization.plots import (
    plot_singular_value_decay,
    plot_dct_basis_functions,
    plot_residual_error_map,
    plot_rate_distortion_curves,
    plot_execution_time_comparison,
    plot_quantized_energy_distribution
)

# Page configuration
st.set_page_config(
    page_title="Math Image Compression | SVD & DCT",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic aesthetic
st.markdown("""
<style>
    .reportview-container { background: #fdfdfd; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1e3d59; margin-bottom: 0px; }
    .subtitle { font-size: 1.05rem; color: #555; margin-bottom: 20px; }
    .math-card { background: #f8f9fa; border-left: 4px solid #1f77b4; padding: 12px 18px; border-radius: 4px; margin-bottom: 15px; }
    .metric-box { background: #ffffff; border: 1px solid #e1e4e8; border-radius: 6px; padding: 10px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { padding-top: 8px; padding-bottom: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_cached_svd(img_array: np.ndarray):
    """Cache SVD decomposition so slider changes only re-evaluate reconstruction."""
    if img_array.ndim == 2:
        U, s, Vt = compute_svd_thin(img_array.astype(np.float64))
        return [U], [s], [Vt]
    else:
        channels = [img_array[:, :, c].astype(np.float64) for c in range(img_array.shape[2])]
        decomp = [compute_svd_thin(ch) for ch in channels]
        Us = [d[0] for d in decomp]
        ss = [d[1] for d in decomp]
        Vts = [d[2] for d in decomp]
        return Us, ss, Vts
def create_image_download(reconstructed_image, filename):
    image_array = to_uint8_image(reconstructed_image)
    image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# Sidebar Configuration
st.sidebar.markdown("##  Image Compression Engine")
# st.sidebar.markdown("**MCA Dissertation Demonstration**")
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigate Module:",
    [
        # "1.  Mathematical Foundations",
        "2.  SVD Low-Rank Studio",
        "3.  8×8 Block DCT Studio",
        "4.  Head-to-Head Comparison",
        "5.  Batch Experimental Suite",
        # "6.  MCA Viva Defense Guide"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Select Input Image")

image_choice_type = st.sidebar.radio("Image Source:", ["Curated Benchmark Library", "Upload Custom Image"])

selected_image_array = None
image_title = "Selected Image"

if image_choice_type == "Curated Benchmark Library":
    sample_files = {
        "Natural Portrait (Smooth gradients)": "natural_portrait.png",
        "Landscape (Horizon & Sky)": "landscape.png",
        "Detailed Texture (High frequency)": "detailed_texture.png",
        "Text Document (Sharp step edges)": "text_document.png",
        "Geometric Shapes (Contrast boundaries)": "geometric_shapes.png"
    }
    sample_choice = st.sidebar.selectbox("Choose Sample:", list(sample_files.keys()))
    sample_path = os.path.join(PROJECT_ROOT, "sample_images", sample_files[sample_choice])
    
    # Process mode: Grayscale or Color
    color_mode = st.sidebar.radio("Processing Representation:", ["Grayscale Matrix (2D)", "RGB Color Tensor (3D)"])
    as_gray = (color_mode == "Grayscale Matrix (2D)")

    if os.path.exists(sample_path):
        selected_image_array, orig_dims = load_image(sample_path, as_grayscale=as_gray)
        image_title = sample_choice
    else:
        st.sidebar.error("Sample image not found. Please run sample generator.")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Image (PNG/JPG/BMP)", type=["png", "jpg", "jpeg", "bmp"])
    color_mode = st.sidebar.radio("Processing Representation:", ["Grayscale Matrix (2D)", "RGB Color Tensor (3D)"])
    as_gray = (color_mode == "Grayscale Matrix (2D)")
    
    if uploaded_file is not None:
        selected_image_array, orig_dims = load_image(uploaded_file, as_grayscale=as_gray)
        image_title = uploaded_file.name
    else:
        # Fallback to default portrait
        default_path = os.path.join(PROJECT_ROOT, "sample_images", "natural_portrait.png")
        if os.path.exists(default_path):
            selected_image_array, orig_dims = load_image(default_path, as_grayscale=as_gray)
            image_title = "Default Natural Portrait"


# ==============================================================================
# MODULE 1: MATHEMATICAL FOUNDATIONS
# ==============================================================================
if app_mode.startswith("1."):
    st.markdown('<p class="main-title">Mathematical Image Compression: Theoretical Foundations</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Linear Algebra, Harmonic Analysis, and Information-Theoretic Storage Models</p>', unsafe_allow_html=True)

    tab_svd, tab_dct, tab_storage, tab_verification = st.tabs([
        "A. Singular Value Decomposition",
        "B. Discrete Cosine Transform",
        "C. Storage Models & Entropy",
        "D. Small Matrix Verification"
    ])

    with tab_svd:
        st.markdown("### 1. Matrix Representation of Images & Fundamental Theorem of SVD")
        st.markdown(r"""
        A digital grayscale image of size $M \times N$ is represented as a real-valued matrix:
        $$A \in \mathbb{R}^{M \times N}, \quad A_{i, j} \in [0, 255]$$
        
        The **Spectral Theorem** guarantees that any real rectangular matrix $A$ can be factored into two orthogonal matrices and a diagonal matrix of singular values:
        """)
        st.latex(r"A = U \Sigma V^T = \sum_{i=1}^{r} \sigma_i u_i v_i^T")
        st.markdown(r"""
        Where:
        - **$U \in \mathbb{R}^{M \times M}$**: Left singular vectors (orthonormal eigenvectors of $A A^T$). $U^T U = I_M$.
        - **$V \in \mathbb{R}^{N \times N}$**: Right singular vectors (orthonormal eigenvectors of $A^T A$). $V^T V = I_N$.
        - **$\Sigma \in \mathbb{R}^{M \times N}$**: Diagonal matrix of singular values in descending order: $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$.
        - **Outer Product Dyads**: $u_i v_i^T \in \mathbb{R}^{M \times N}$ is a rank-1 orthogonal basis image.
        """)

        st.markdown("### 2. Low-Rank Approximation & The Eckart-Young-Mirsky Theorem")
        st.markdown(r"""
        By selecting a truncation parameter $k \le \text{rank}(A)$, we form the rank-$k$ approximation:
        """)
        st.latex(r"A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T")
        st.markdown(r"""
        > **Eckart-Young-Mirsky Theorem (1936):**  
        > For any matrix $B$ of rank at most $k$, the rank-$k$ truncated SVD matrix $A_k$ is the globally optimal approximation under both Frobenius and Spectral norms:
        """)
        st.latex(r"\min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{\min(M, N)} \sigma_i^2}")
        st.latex(r"\min_{\text{rank}(B) \le k} \|A - B\|_2 = \|A - A_k\|_2 = \sigma_{k+1}")

    with tab_dct:
        st.markdown("### 1. 2D Discrete Cosine Transform (DCT-II)")
        st.markdown(r"""
        While SVD computes data-dependent bases ($u_i, v_i$), DCT uses **fixed, data-independent orthonormal sinusoidal basis functions**.
        For an $8 \times 8$ pixel block $f(x, y)$ ($x, y \in \{0, \dots, 7\}$), the forward 2D DCT-II is:
        """)
        st.latex(r"F(u, v) = \frac{1}{4} C(u) C(v) \sum_{x=0}^{7} \sum_{y=0}^{7} f(x, y) \cos\left[ \frac{(2x + 1)u \pi}{16} \right] \cos\left[ \frac{(2y + 1)v \pi}{16} \right]")
        st.markdown(r"""
        where $C(0) = \frac{1}{\sqrt{2}}$ and $C(u) = 1$ for $u > 0$.
        
        Using matrix operators, let $T \in \mathbb{R}^{8 \times 8}$ be the orthonormal DCT matrix. Then:
        - **Forward DCT**: $F = T (X - 128) T^T$
        - **Inverse DCT (IDCT)**: $\hat{X} = T^T F T + 128$
        """)

        st.markdown("### 2. Psychovisual Quantization")
        st.markdown(r"""
        Compression is introduced in the quantization stage, taking advantage of the Human Visual System (HVS) insensitivity to high spatial frequencies:
        """)
        st.latex(r"F_Q(u, v) = \text{round}\left( \frac{F(u, v)}{Q(u, v) \cdot S(q)} \right)")
        st.latex(r"\hat{F}(u, v) = F_Q(u, v) \times [Q(u, v) \cdot S(q)]")

        st.subheader("8×8 DCT Basis Function Dictionary")
        basis_fig = plot_dct_basis_functions(8)
        st.pyplot(basis_fig)

    with tab_storage:
        st.markdown("### Theoretical Storage Models: Distinction Between Parameters and File Size")
        st.markdown(r"""
        A common pitfall in computer applications projects is treating mathematical low-rank factors or quantized coefficients as ready-to-store disk files.
        We strictly distinguish:
        """)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### SVD Storage Model")
            st.markdown(r"""
            - **Raw Image Parameters**: $M \times N$ values
            - **Truncated SVD Parameters**: $k(M + N + 1)$ values
              - $U_k$: $M \times k$
              - $\Sigma_k$: $k$
              - $V_k^T$: $k \times N$
            - **Theoretical Compression Ratio**:
            """)
            st.latex(r"CR_{\text{SVD}} = \frac{M \times N}{k(M + N + 1)}")
            st.markdown(r"""
            - **Critical Breakeven Rank ($k^*$)**:
            """)
            st.latex(r"k^* = \left\lfloor \frac{M \times N}{M + N + 1} \right\rfloor \approx \frac{N}{2} \text{ (for square images)}")
            st.warning("⚠️ If $k > k^*$, SVD requires MORE stored parameters than the uncompressed image!")

        with c2:
            st.markdown("#### DCT Entropy Storage Model")
            st.markdown(r"""
            - **Total Coefficients**: $M \times N$
            - After quantization, high frequencies collapse to zero (inducing high sparsity).
            - **First-Order Shannon Entropy**:
            """)
            st.latex(r"\mathcal{H}(F_Q) = - \sum_{i=1}^{L} p(s_i) \log_2 p(s_i) \quad [\text{bits/coefficient}]")
            st.markdown(r"""
            - **Theoretical Compression Ratio**:
            """)
            st.latex(r"CR_{\text{DCT}} = \frac{8.0}{\mathcal{H}(F_Q)}")
            st.info("ℹ️ In a real JPEG file, Huffman coding and RLE pack these symbols into bytes, plus metadata headers.")

    with tab_verification:
        st.markdown("### Small-Matrix Numerical Verification")
        st.markdown("Validation of algorithms against hand-verifiable matrices to eliminate black-box claims.")
        
        c_svd_v, c_dct_v = st.columns(2)
        with c_svd_v:
            st.markdown("#### SVD 4×4 Rank-2 Matrix Test")
            svd_test = verify_svd_on_small_matrix()
            st.json(svd_test)
            if svd_test["eckart_young_verified"]:
                st.success("✅ Eckart-Young Theorem analytically verified on test matrix!")

        with c_dct_v:
            st.markdown("#### DCT 8×8 Invertibility & Basis Test")
            dct_test = verify_dct_on_small_block()
            st.json(dct_test)
            if dct_test["is_orthonormal"] and dct_test["is_invertible"]:
                st.success("✅ Orthonormality ($T T^T = I$) and perfect inversion ($<10^{-11}$) verified!")


# ==============================================================================
# MODULE 2: SVD LOW-RANK COMPRESSION STUDIO
# ==============================================================================
elif app_mode.startswith("2."):
    st.markdown('<p class="main-title">SVD Low-Rank Compression Studio</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Active Target: <b>{image_title}</b> | Dimensions: <b>{selected_image_array.shape[0]}×{selected_image_array.shape[1]}</b></p>', unsafe_allow_html=True)

    H, W = selected_image_array.shape[:2]
    max_k = min(H, W)

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        default_k = max(1, min(25, max_k))
        k = st.slider("Select Approximation Rank $k$ (Retained Singular Values):", min_value=1, max_value=max_k, value=default_k, step=1)
    with col_ctrl2:
        breakeven = int((H * W) / (H + W + 1))
        st.info(f"**Theoretical Breakeven Rank:** $k^* = {breakeven}$\n\n*(Values above {breakeven} expand parameter count)*")

    # Perform compression
    with st.spinner("Computing Low-Rank Reconstruction..."):
        recon_svd, meta_svd = compress_svd_image(selected_image_array, k=k)
        metrics_svd = compute_all_metrics(selected_image_array.astype(np.float64), recon_svd)
        storage_svd = analyze_svd_storage(selected_image_array.shape, k=k)

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Approximation Rank", f"k = {k}", f"Max: {max_k}")
    m2.metric("Energy Retained", f"{meta_svd['energy_retained_percent']:.2f}%", f"{meta_svd['energy_retained_percent'] - 100:.2f}%")
    m3.metric("PSNR", f"{metrics_svd['psnr_db']:.2f} dB")
    m4.metric("MSE", f"{metrics_svd['mse']:.2f}")
    m5.metric("Param CR", f"{storage_svd['parameter_compression_ratio']:.2f} : 1")
    m6.metric("Total Time", f"{meta_svd['total_time_sec']*1000.0:.1f} ms")

    # Download compressed SVD image
    # Download compressed SVD image
    svd_download = create_image_download(
    recon_svd,
    "svd_compressed.png"
)

    st.download_button(
    label="Download SVD Compressed Image",
    data=svd_download,
    file_name="svd_compressed.png",
    mime="image/png",
    use_container_width=True
)

    # Visual Inspection
    st.markdown("### Visual Comparison")

    c_img1, c_img2, c_img3 = st.columns(3)
    with c_img1:
        st.markdown("**Original Image**")
        st.image(selected_image_array, clamp=True, use_container_width=True)
    with c_img2:
        st.markdown(f"**SVD Reconstructed ($k={k}$)**")
        st.image(to_uint8_image(recon_svd), clamp=True, use_container_width=True)
    with c_img3:
        st.markdown(r"**Residual Error Map ($|I - \hat{I}|$)**")
        fig_err = plot_residual_error_map(selected_image_array, recon_svd, title=f"SVD Residual Error (Rank {k})")
        st.pyplot(fig_err)

    # Singular Value Spectrum & Energy
    st.markdown("### Singular Value Spectrum & Energy Decay")
    fig_scree = plot_singular_value_decay(meta_svd["singular_values"], k_selected=k)
    st.pyplot(fig_scree)


# ==============================================================================
# MODULE 3: 8x8 BLOCK DCT COMPRESSION STUDIO
# ==============================================================================
elif app_mode.startswith("3."):
    st.markdown('<p class="main-title">8×8 Block DCT Compression Studio</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Active Target: <b>{image_title}</b> | Dimensions: <b>{selected_image_array.shape[0]}×{selected_image_array.shape[1]}</b></p>', unsafe_allow_html=True)

    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        quality = st.slider("Select Quality Factor $q$ (1 = High Quantization, 100 = Negligible Loss):", min_value=1, max_value=100, value=50, step=1)
    with col_ctrl2:
        st.info("Quality factor scales the standard JPEG Luminance table using the IJG formula.")

    # Perform compression
    with st.spinner("Transforming via 8x8 Block DCT..."):
        recon_dct, meta_dct = compress_dct_image(selected_image_array, quality=quality)
        metrics_dct = compute_all_metrics(selected_image_array.astype(np.float64), recon_dct)
        storage_dct = analyze_dct_storage(
            selected_image_array.shape,
            meta_dct["quantized_coefficients"],
            meta_dct["shannon_entropy_bpc"]
        )

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Quality Factor", f"q = {quality}")
    m2.metric("Quantized Sparsity", f"{storage_dct['sparsity_percent']:.1f}%", "Zero Coeffs")
    m3.metric("PSNR", f"{metrics_dct['psnr_db']:.2f} dB")
    m4.metric("MSE", f"{metrics_dct['mse']:.2f}")
    m5.metric("Entropy CR", f"{storage_dct['theoretical_compression_ratio']:.2f} : 1")
    m6.metric("Execution Time", f"{meta_dct['total_time_sec']*1000.0:.1f} ms")
    # Download compressed DCT image
    dct_download = create_image_download(
    recon_dct,
    "dct_compressed.png"
)

    st.download_button(
    label="Download DCT Compressed Image",
    data=dct_download,
    file_name="dct_compressed.png",
    mime="image/png",
    use_container_width=True
)

    # Visual Inspection
    st.markdown("### Visual Comparison")
    c_img1, c_img2, c_img3 = st.columns(3)
    with c_img1:
        st.markdown("**Original Image**")
        st.image(selected_image_array, clamp=True, use_container_width=True)
    with c_img2:
        st.markdown(f"**DCT Reconstructed ($q={quality}$)**")
        st.image(to_uint8_image(recon_dct), clamp=True, use_container_width=True)
    with c_img3:
        st.markdown(r"**Residual Error Map ($|I - \hat{I}|$)**")
        fig_err_dct = plot_residual_error_map(selected_image_array, recon_dct, title=f"DCT Residual Error (Quality {quality})")
        st.pyplot(fig_err_dct)

    # Energy Compaction into 8x8 Frequency Blocks
    st.markdown("### Quantized Coefficient Energy Distribution")
    c_hm1, c_hm2 = st.columns([1, 1])
    with c_hm1:
        fig_heat = plot_quantized_energy_distribution(meta_dct["quantized_coefficients"])
        st.pyplot(fig_heat)
    with c_hm2:
        st.markdown("#### Frequency Compaction Analysis")
        st.markdown(f"""
        - **Total 8×8 Blocks Processed**: `{(meta_dct['padded_dims'][0]//8) * (meta_dct['padded_dims'][1]//8):,}`
        - **Total Transform Coefficients**: `{storage_dct['total_coefficients']:,}`
        - **Zero Coefficients (Truncated)**: `{storage_dct['zero_coefficients']:,}`
        - **Non-Zero Coefficients Retained**: `{storage_dct['non_zero_coefficients']:,}`
        - **First-Order Shannon Entropy**: `{storage_dct['shannon_entropy_bpc']:.3f} bits/coefficient`
        - **Uncompressed Baseline**: `8.000 bits/pixel`
        
        Notice how energy clusters heavily in the top-left DC coefficient ($u=0, v=0$), while higher spatial frequencies along the bottom-right diagonals round to exactly 0.0.
        """)


# ==============================================================================
# MODULE 4: HEAD-TO-HEAD COMPARISON
# ==============================================================================
elif app_mode.startswith("4."):
    st.markdown('<p class="main-title">Head-to-Head Comparative Dashboard</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Direct Empirical Benchmark: SVD vs DCT on <b>{image_title}</b></p>', unsafe_allow_html=True)

    H, W = selected_image_array.shape[:2]
    max_k = min(H, W)

    c_param1, c_param2 = st.columns(2)
    with c_param1:
        k_comp = st.slider("Select SVD Rank $k$:", min_value=1, max_value=max_k, value=max(1, min(20, max_k)))
    with c_param2:
        q_comp = st.slider("Select DCT Quality Factor $q$:", min_value=1, max_value=100, value=50)

    # Run both compressions
    recon_svd, meta_svd = compress_svd_image(selected_image_array, k=k_comp)
    recon_dct, meta_dct = compress_dct_image(selected_image_array, quality=q_comp)

    metrics_svd = compute_all_metrics(selected_image_array.astype(np.float64), recon_svd)
    metrics_dct = compute_all_metrics(selected_image_array.astype(np.float64), recon_dct)

    storage_svd = analyze_svd_storage(selected_image_array.shape, k=k_comp)
    storage_dct = analyze_dct_storage(selected_image_array.shape, meta_dct["quantized_coefficients"], meta_dct["shannon_entropy_bpc"])

    # Side-by-Side Images
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Original Image**")
        st.image(selected_image_array, clamp=True, use_container_width=True)
    with c2:
        st.markdown(f"**SVD (Rank $k={k_comp}$)**")
        st.image(to_uint8_image(recon_svd), clamp=True, use_container_width=True)
    with c3:
        st.markdown(f"**DCT (Quality $q={q_comp}$)**")
        st.image(to_uint8_image(recon_dct), clamp=True, use_container_width=True)

    # Comparative Metrics Table
    st.markdown("### Comparative Performance Evaluation Table")
    comp_df = pd.DataFrame({
        "Metric / Property": [
            "Mathematical Foundation",
            "Mean Squared Error (MSE) ↓",
            "Root Mean Squared Error (RMSE) ↓",
            "Mean Absolute Error (MAE) ↓",
            "Peak Signal-to-Noise Ratio (PSNR) ↑",
            "Structural Similarity (SSIM) ↑",
            "Theoretical Compression Ratio (CR) ↑",
            "Space Saving Percentage ↑",
            "Execution Time (ms) ↓",
            "Characteristic Visual Artifacts"
        ],
        "Singular Value Decomposition (SVD)": [
            "Data-Dependent Eigenvector Factorization",
            f"{metrics_svd['mse']:.2f}",
            f"{metrics_svd['rmse']:.2f}",
            f"{metrics_svd['mae']:.2f}",
            f"{metrics_svd['psnr_db']:.2f} dB",
            f"{metrics_svd['ssim']:.4f}",
            f"{storage_svd['parameter_compression_ratio']:.2f} : 1 (Param)",
            f"{storage_svd['parameter_space_saving_percent']:.1f}%",
            f"{meta_svd['total_time_sec']*1000.0:.2f} ms",
            "Low-rank blurring, loss of texture, ghosting"
        ],
        "Discrete Cosine Transform (DCT)": [
            "Fixed Orthonormal Cosine Basis (8×8 Blocks)",
            f"{metrics_dct['mse']:.2f}",
            f"{metrics_dct['rmse']:.2f}",
            f"{metrics_dct['mae']:.2f}",
            f"{metrics_dct['psnr_db']:.2f} dB",
            f"{metrics_dct['ssim']:.4f}",
            f"{storage_dct['theoretical_compression_ratio']:.2f} : 1 (Entropy)",
            f"{storage_dct['entropy_space_saving_percent']:.1f}%",
            f"{meta_dct['total_time_sec']*1000.0:.2f} ms",
            "8×8 blocking boundary discontinuities, Gibbs ringing"
        ]
    })
    st.table(comp_df)

    # Automated Sweep Curves for Current Image
    st.markdown("### Dynamic Rate-Distortion Curves for Selected Image")
    with st.spinner("Computing parameter sweeps for Rate-Distortion curve..."):
        svd_sweep = benchmark_svd_ranks(selected_image_array)
        dct_sweep = benchmark_dct_qualities(selected_image_array)
        fig_rd = plot_rate_distortion_curves(svd_sweep, dct_sweep)
        st.pyplot(fig_rd)


# ==============================================================================
# MODULE 5: BATCH EXPERIMENTAL SUITE
# ==============================================================================
elif app_mode.startswith("5."):
    st.markdown('<p class="main-title">Batch Experimental Benchmark & Dataset Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Systematic Reproducible Experiments Across Diverse Image Categories</p>', unsafe_allow_html=True)

    csv_path = os.path.join(PROJECT_ROOT, "experiments", "experimental_results.csv")
    fig_dir = os.path.join(PROJECT_ROOT, "experiments", "figures")

    if st.button("▶️ Run Fresh Experimental Suite Now"):
        with st.spinner("Executing full experimental suite across all sample images..."):
            from experiments.run_experiments import run_full_experimental_suite
            df_fresh = run_full_experimental_suite()
            st.success("Completed 85 trials successfully! Updated dataset and figures.")

    if os.path.exists(csv_path):
        df_exp = pd.read_csv(csv_path)
        st.markdown(f"**Loaded Experimental Trials:** `{len(df_exp)} runs` across `{df_exp['Image_Name'].nunique()} image classes`")

        tab_data, tab_figs = st.tabs(["📊 Experimental Dataset Table", "📈 Published Comparative Graphs"])

        with tab_data:
            st.dataframe(df_exp, use_container_width=True)
            csv_bytes = df_exp.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Experimental Dataset (CSV)",
                data=csv_bytes,
                file_name="svd_dct_experimental_results.csv",
                mime="text/csv"
            )

        with tab_figs:
            fig1_path = os.path.join(fig_dir, "fig1_rate_distortion_all_images.png")
            fig2_path = os.path.join(fig_dir, "fig2_mse_comparison.png")
            fig3_path = os.path.join(fig_dir, "fig3_execution_time_comparison.png")
            fig4_path = os.path.join(fig_dir, "fig4_svd_rank_effect.png")
            fig5_path = os.path.join(fig_dir, "fig5_dct_quality_effect.png")

            if os.path.exists(fig1_path):
                st.image(fig1_path, caption="Figure 1: Rate-Distortion Curves (PSNR vs CR) Across Diverse Images", use_container_width=True)
            if os.path.exists(fig2_path):
                st.image(fig2_path, caption="Figure 2: Mean Squared Error (MSE) vs Compression Ratio", use_container_width=True)
            if os.path.exists(fig3_path):
                st.image(fig3_path, caption="Figure 3: Computational Execution Time Comparison", use_container_width=True)
            if os.path.exists(fig4_path):
                st.image(fig4_path, caption="Figure 4: Effect of Truncated Rank k on SVD Reconstruction", use_container_width=True)
            if os.path.exists(fig5_path):
                st.image(fig5_path, caption="Figure 5: Effect of Quality Factor q on Quantization Sparsity", use_container_width=True)
    else:
        st.warning("No experimental dataset found. Please click 'Run Fresh Experimental Suite Now' to generate.")


# ==============================================================================
# MODULE 6: MCA VIVA DEFENSE GUIDE
# ==============================================================================
elif app_mode.startswith("6."):
    st.markdown('<p class="main-title">MCA Project Viva Examination & Theory Guide</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Comprehensive Q&A, Mathematical Derivations, and Defense Preparation</p>', unsafe_allow_html=True)

    viva_qas = [
        {
            "q": "1. What is the fundamental mathematical difference between SVD and DCT compression?",
            "a": r"""
            **SVD uses data-dependent basis functions**, derived directly from the eigenvectors of $A A^T$ and $A^T A$. It produces the mathematically optimal low-rank approximation for that specific matrix, but the basis vectors $U_k$ and $V_k$ must be stored alongside singular values $\Sigma_k$.
            
            **DCT uses fixed, data-independent basis functions** (cosinusoids). The decoder already knows the basis functions, so only the quantized transform coefficients need to be transmitted/stored.
            """
        },
        {
            "q": "2. Why does SVD rarely produce a smaller image file on disk in real-world software?",
            "a": r"""
            This is a critical distinction:
            - An 8-bit uncompressed grayscale image requires $M \times N \times 1$ byte.
            - Low-rank SVD retains $k(M + N + 1)$ **floating-point values** (typically 32-bit float = 4 bytes or 64-bit float = 8 bytes).
            - Therefore, in bytes: $\text{Storage}_{\text{SVD}} = 4 \cdot k(M + N + 1)$.
            - For SVD to achieve actual byte-level compression ($< M \cdot N$), rank $k$ must satisfy:
              $$k < \frac{M \cdot N}{4(M + N + 1)}$$
            For a $512 \times 512$ image, $k$ must be less than $\approx 64$. At $k=64$, visual quality is often degraded. SVD is thus primarily a **mathematical feature extraction and dimensionality reduction tool**, not a standalone general-purpose file codec.
            """
        },
        {
            "q": "3. State the Eckart-Young-Mirsky Theorem and explain its importance.",
            "a": r"""
            **Theorem:** For any target rank $k \le \text{rank}(A)$, the truncated SVD matrix $A_k = U_k \Sigma_k V_k^T$ minimizes the Frobenius norm distance:
            $$\min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{\min(M, N)} \sigma_i^2}$$
            **Significance:** It mathematically proves that no other rank-$k$ factorization (e.g. PCA, QR, LU, NMF) can achieve a lower Mean Squared Error for the same rank $k$.
            """
        },
        {
            "q": "4. Why is the image partitioned into 8×8 blocks in DCT instead of computing DCT for the whole image at once?",
            "a": r"""
            Three primary reasons:
            1. **Computational Complexity**: Full-image DCT requires $\mathcal{O}(M N \log(MN))$ operations. Block DCT processes $\frac{MN}{64}$ blocks of size $8 \times 8$, requiring $\mathcal{O}(MN)$ total operations, which is linear and fast.
            2. **Local Correlation**: Spatial correlation in natural images is highly localized. Pixels 500 pixels apart have near-zero correlation; computing global transform spreads localized noise everywhere (ringing artifacts).
            3. **Memory & Streaming**: $8 \times 8$ blocks fit cleanly inside CPU L1 cache, allowing hardware acceleration.
            """
        },
        {
            "q": "5. What are the characteristic visual artifacts of SVD vs DCT at high compression ratios?",
            "a": r"""
            - **SVD**: Produces **directional blurring and geometric smearing**. Because it retains only dominant principal dyads, fine high-frequency textures disappear, and high-contrast lines look like faint low-rank 'ghosts'.
            - **DCT**: Produces **blocking artifacts** (visible grid lines along the $8 \times 8$ block borders due to independent block quantization) and **Gibbs ringing** around sharp step edges (like text typography).
            """
        },
        {
            "q": "6. Why is PSNR defined on a logarithmic (decibel) scale rather than just reporting MSE?",
            "a": r"""
            The Human Visual System (HVS) perceives brightness and distortion logarithmically (Weber-Fechner Law). A difference of 100 in MSE between 1000 and 1100 is visually unnoticeable, whereas a difference of 100 between 1 and 101 represents a catastrophic collapse in image fidelity. PSNR maps MSE into a decibel scale ($20 \log_{10} \frac{255}{\sqrt{\text{MSE}}}$), which aligns closely with perceived visual degradation.
            """
        }
    ]

    for item in viva_qas:
        with st.expander(item["q"], expanded=True):
            st.markdown(item["a"])

st.markdown("---")
st.markdown("<div style='text-align: center; color: #888; font-size: 0.85rem;'>Master of Computer Applications (MCA) Project | Mathematical Image Compression Using SVD and DCT | Built with Python, NumPy, SciPy & Streamlit</div>", unsafe_allow_html=True)
