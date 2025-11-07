import io
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------
# Utilities
# ---------------------------------------------

def load_image_to_bgr(file_bytes) -> np.ndarray:
    file_bytes = np.asarray(bytearray(file_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Please upload a valid image file.")
    return img

@st.cache_data(show_spinner=False)
def to_gray_cached(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

@st.cache_data(show_spinner=False)
def to_rgb_cached(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def normalize01(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx <= mn:
        return np.zeros_like(a, dtype=np.float32)
    return (a - mn) / (mx - mn)


def apply_heatmap01(x01: np.ndarray) -> np.ndarray:
    """Input 0..1 float -> heatmap BGR uint8."""
    x255 = np.clip(x01 * 255.0, 0, 255).astype(np.uint8)
    hm = cv2.applyColorMap(x255, cv2.COLORMAP_JET)
    return hm


def overlay_mask_on_image(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.5,
                          color=(0, 0, 255)) -> np.ndarray:
    # Robust overlay that tolerates 1- or 3-channel masks and mismatched sizes
    if mask is None:
        return bgr
    if mask.ndim == 3:
        # Reduce to single channel if someone passed a color mask
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Ensure mask matches the image size
    if bgr.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    # Binary mask as boolean
    m = (mask > 0)
    # Create a colored layer and blend only where mask is True
    overlay = bgr.copy()
    color_img = np.zeros_like(overlay)
    color_img[:] = color
    overlay[m] = color_img[m]
    out = cv2.addWeighted(bgr, 1 - alpha, overlay, alpha, 0)
    return out


def bin_kernel(k: int) -> np.ndarray:
    k = max(1, int(k))
    return np.ones((k, k), np.uint8)


def oddify(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def download_png_button(image_bgr: np.ndarray, label: str, key: str):
    ok, buf = cv2.imencode('.png', image_bgr)
    if not ok:
        st.warning("Could not encode image for download.")
        return
    st.download_button(label, data=buf.tobytes(), file_name=f"{key}.png", mime="image/png", key=f"dl_{key}")

# ---------------------------------------------
# Methods
# ---------------------------------------------

# 1) Simple Threshold on intensity/gradient with morphology

def method_threshold(gray: np.ndarray,
                     threshold: int,
                     use_gradient: bool,
                     blur_ksize: int,
                     close_ksize: int,
                     alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    src = gray
    if use_gradient:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        src = np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    if blur_ksize > 0:
        k = oddify(blur_ksize)
        src = cv2.GaussianBlur(src, (k, k), 0)

    _, bin_img = cv2.threshold(src, threshold, 255, cv2.THRESH_BINARY)
    if close_ksize > 0:
        kernel = bin_kernel(close_ksize)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    heatmap = apply_heatmap01(normalize01(src))
    return heatmap, bin_img, heatmap_overlay(heatmap, bin_img, alpha)


def heatmap_overlay(heatmap_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    # Guarantee heatmap is 3-channel BGR
    if heatmap_bgr.ndim == 2:
        base = cv2.cvtColor(heatmap_bgr, cv2.COLOR_GRAY2BGR)
    else:
        base = heatmap_bgr.copy()
    return overlay_mask_on_image(base, mask, alpha=alpha, color=(0, 0, 255))


# 2) Canny + fill largest region + dilation

def fill_largest_from_edges(edges: np.ndarray) -> np.ndarray:
    # Find contours on edges and fill the largest area after closing small gaps
    kernel = bin_kernel(3)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(edges.shape, np.uint8)
    if not contours:
        return mask
    areas = [cv2.contourArea(c) for c in contours]
    idx = int(np.argmax(areas))
    cv2.drawContours(mask, [contours[idx]], -1, 255, thickness=cv2.FILLED)
    return mask


def method_canny_fill(bgr: np.ndarray, gray: np.ndarray,
                      t_low: int, t_high: int,
                      blur_ksize: int,
                      dilate_iters: int,
                      alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gray
    if blur_ksize > 0:
        k = oddify(blur_ksize)
        g = cv2.GaussianBlur(gray, (k, k), 0)

    edges = cv2.Canny(g, t_low, t_high)
    filled = fill_largest_from_edges(edges)
    if dilate_iters > 0:
        kernel = bin_kernel(3)
        filled = cv2.dilate(filled, kernel, iterations=dilate_iters)

    heatmap = apply_heatmap01(normalize01(g))
    overlay = overlay_mask_on_image(bgr, filled, alpha=alpha, color=(0, 0, 255))
    return heatmap, filled, overlay


# 3) CLAHE + Otsu (with bias) or Adaptive + morphology

def apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clip_limit)), tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def method_intensity(gray: np.ndarray,
                     blur_ksize: int,
                     use_clahe: bool,
                     clahe_clip: float,
                     clahe_grid: int,
                     mode: str,
                     otsu_bias: float,
                     adapt_method: str,
                     adapt_block: int,
                     adapt_c: int,
                     morph_ksize: int,
                     morph_iters: int,
                     alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gray
    if use_clahe:
        g = apply_clahe(g, clahe_clip, clahe_grid)
    if blur_ksize > 0:
        k = oddify(blur_ksize)
        g = cv2.GaussianBlur(g, (k, k), 0)

    if mode == "Otsu":
        # First, plain Otsu to get threshold, then apply bias
        _t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        biased_t = max(0, min(255, int(_t * float(otsu_bias))))
        _, bin_img = cv2.threshold(g, biased_t, 255, cv2.THRESH_BINARY)
    else:
        am = cv2.ADAPTIVE_THRESH_MEAN_C if adapt_method == "Mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        k = oddify(adapt_block)
        bin_img = cv2.adaptiveThreshold(g, 255, am, cv2.THRESH_BINARY, k, adapt_c)

    if morph_ksize > 0 and morph_iters > 0:
        kernel = bin_kernel(morph_ksize)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=morph_iters)

    heatmap = apply_heatmap01(normalize01(g))
    overlay = heatmap_overlay(heatmap, bin_img, alpha)
    return heatmap, bin_img, overlay


# 4) Bright-band guided ROI + segmentation

def longest_run(mask1d: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return [y0, y1] inclusive for the longest True run in a 1-D mask.
    Robust to empty inputs, all-False, and odd dtypes.
    """
    if mask1d is None:
        return None
    if mask1d.ndim != 1:
        mask1d = mask1d.ravel()
    if mask1d.size == 0:
        return None
    # Ensure boolean 0/1
    x = (mask1d.astype(np.float32) > 0).astype(np.uint8)
    # Pad with zeros so every run has a start and an end
    xpad = np.r_[0, x, 0]
    d = np.diff(xpad)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return None
    # Guard against any mismatch (shouldn't happen with padding, but be safe)
    n = min(starts.size, ends.size)
    if n == 0:
        return None
    starts = starts[:n]
    ends = ends[:n]
    lengths = ends - starts
    if lengths.size == 0:
        return None
    k = int(np.argmax(lengths))
    return int(starts[k]), int(ends[k] - 1)


def method_bright_band(bgr: np.ndarray, gray: np.ndarray,
                       use_clahe: bool,
                       clahe_clip: float,
                       clahe_grid: int,
                       bilateral_d: int,
                       bilateral_sigma: int,
                       profile_pct: int,
                       band_frac_of_peak: float,
                       band_margin: int,
                       min_area_frac: float,
                       min_aspect: float,
                       alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    g = gray
    if bilateral_d > 0:
        g = cv2.bilateralFilter(g, d=bilateral_d, sigmaColor=bilateral_sigma, sigmaSpace=bilateral_sigma)
    if use_clahe:
        g = apply_clahe(g, clahe_clip, clahe_grid)

    H, W = g.shape
    # Row-wise high-percentile profile
    q = np.percentile(g, profile_pct, axis=1).astype(np.float32)
    q_s = cv2.GaussianBlur(q.reshape(-1, 1), (1, 21), 0).ravel()
    peak = float(q_s.max()) + 1e-6
    band_mask_1d = q_s >= band_frac_of_peak * peak

    run = longest_run(band_mask_1d)
    if run is None:
        roi_mask = np.ones_like(g, np.uint8) * 255  # fall back to full image
    else:
        y0, y1 = run
        y0 = max(0, y0 - band_margin)
        y1 = min(H - 1, y1 + band_margin)
        roi_mask = np.zeros_like(g, np.uint8)
        roi_mask[y0:y1 + 1, :] = 255

    # Otsu inside the band ROI
    g_roi = cv2.bitwise_and(g, g, mask=roi_mask)
    _t, _ = cv2.threshold(g_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_img = cv2.threshold(g, int(_t), 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=roi_mask)

    # Keep the largest horizontally elongated component satisfying constraints
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filt = np.zeros_like(bin_img)
    best_area = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area_frac * (H * W):
            continue
        aspect = (w / (h + 1e-6))
        if aspect < min_aspect:
            continue
        if area > best_area:
            best_area = area
            filt[:] = 0
            cv2.drawContours(filt, [c], -1, 255, thickness=cv2.FILLED)

    heatmap = apply_heatmap01(normalize01(g))
    overlay = overlay_mask_on_image(bgr, filt, alpha=alpha, color=(0, 0, 255))
    return heatmap, filt, overlay


# ---------------------------------------------
# UI
# ---------------------------------------------

st.set_page_config(page_title="Weld Bead Detection Lab", layout="wide")
st.title("Weld Bead Detection Lab")

st.write(
    "Upload a single, **already-cropped** weld image. Explore alternative detection methods in tabs, and tune parameters with sliders."
)

uploaded = st.file_uploader("Upload weld image (PNG/JPG/BMP, cropped)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if uploaded is None:
    st.info("⬆️ Upload an image to get started.")
    st.stop()

bgr = load_image_to_bgr(uploaded)
rgb = to_rgb_cached(bgr)
gray = to_gray_cached(bgr)

st.subheader("Preview")
st.image(rgb, caption=f"Input ({bgr.shape[1]}×{bgr.shape[0]})", use_column_width=True)

# Tabs for different methods
TAB_TITLES = [
    "Threshold (Intensity/Gradient)",
    "Canny + Fill Largest",
    "CLAHE + Otsu / Adaptive",
    "Bright-Band Guided"
]

tab1, tab2, tab3, tab4 = st.tabs(TAB_TITLES)

with tab1:
    st.markdown("### 1) Threshold (Intensity or Gradient)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        use_gradient = st.toggle("Use gradient magnitude", value=False)
    with c2:
        threshold = st.slider("Threshold", min_value=0, max_value=255, value=140, step=1)
    with c3:
        blur_ksize = st.slider("Gaussian blur ksize", 0, 21, 5, step=1)
    with c4:
        close_ksize = st.slider("Morph close ksize", 0, 21, 5, step=1)
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05)

    heatmap, binary, overlay = method_threshold(gray, threshold, use_gradient, blur_ksize, close_ksize, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Intensity/Gradient Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary Weld Region")
        st.image(binary, use_column_width=True, clamp=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.expander("Download results"):
        download_png_button(heatmap, "Download Heatmap", key="th_heatmap")
        download_png_button(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Download Binary", key="th_binary")
        download_png_button(overlay, "Download Overlay", key="th_overlay")

with tab2:
    st.markdown("### 2) Canny + Fill Largest Region")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        t_low = st.slider("Canny low", 0, 255, 10, step=1)
    with c2:
        ratio = st.select_slider("High/Low ratio", options=[2.0, 2.5, 3.0, 3.5], value=3.0)
        t_high = int(t_low * float(ratio))
        t_high = max(t_low + 1, min(255, t_high))
        st.write(f"High ≈ {t_high}")
    with c3:
        blur_ksize = st.slider("Pre-blur ksize", 0, 21, 5, step=1)
    with c4:
        dilate_iters = st.slider("Dilate iterations", 0, 5, 2)
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="alpha_canny")

    heatmap, filled, overlay = method_canny_fill(bgr, gray, t_low, t_high, blur_ksize, dilate_iters, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Preprocessed Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Filled Largest Region")
        st.image(filled, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.expander("Download results"):
        download_png_button(heatmap, "Download Heatmap", key="canny_heatmap")
        download_png_button(cv2.cvtColor(filled, cv2.COLOR_GRAY2BGR), "Download Binary", key="canny_binary")
        download_png_button(overlay, "Download Overlay", key="canny_overlay")

with tab3:
    st.markdown("### 3) CLAHE + Otsu / Adaptive Thresholding")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        use_clahe = st.toggle("Enable CLAHE", value=True)
    with c2:
        clahe_clip = st.slider("CLAHE clipLimit", 0.1, 5.0, 2.0, step=0.1)
    with c3:
        clahe_grid = st.slider("CLAHE tileGridSize", 4, 32, 8, step=1)
    with c4:
        blur_ksize = st.slider("Gaussian blur ksize", 0, 21, 5, step=1, key="int_blur")
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="alpha_int")

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        mode = st.radio("Threshold mode", ["Otsu", "Adaptive"], horizontal=True)
    with c7:
        otsu_bias = st.slider("Otsu bias (× T)", 0.5, 1.5, 1.0, step=0.05)
    with c8:
        adapt_method = st.selectbox("Adaptive method", ["Mean", "Gaussian"], index=1)
    with c9:
        adapt_block = st.slider("Adaptive block size (odd)", 3, 51, 21, step=2)
    c10, c11 = st.columns(2)
    with c10:
        adapt_c = st.slider("Adaptive C (stricter when higher)", -10, 20, 2, step=1)
    with c11:
        morph_ksize = st.slider("Morph open ksize", 0, 21, 5, step=1)
    morph_iters = st.slider("Morph open iters", 0, 5, 1)

    heatmap, binary, overlay = method_intensity(
        gray=gray,
        blur_ksize=blur_ksize,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        mode=mode,
        otsu_bias=otsu_bias,
        adapt_method=adapt_method,
        adapt_block=adapt_block,
        adapt_c=adapt_c,
        morph_ksize=morph_ksize,
        morph_iters=morph_iters,
        alpha=alpha,
    )

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Intensity Heatmap (preprocessed)")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary Weld Region")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.expander("Download results"):
        download_png_button(heatmap, "Download Heatmap", key="int_heatmap")
        download_png_button(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Download Binary", key="int_binary")
        download_png_button(overlay, "Download Overlay", key="int_overlay")

with tab4:
    st.markdown("### 4) Bright-Band Guided Detection")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        use_clahe = st.toggle("Enable CLAHE", value=True, key="bb_clahe")
    with c2:
        clahe_clip = st.slider("CLAHE clipLimit", 0.1, 5.0, 2.0, step=0.1, key="bb_clip")
    with c3:
        clahe_grid = st.slider("CLAHE tileGridSize", 4, 32, 8, step=1, key="bb_grid")
    with c4:
        bilateral_d = st.slider("Bilateral d", 0, 15, 5, step=1)
    with c5:
        bilateral_sigma = st.slider("Bilateral sigma (color/space)", 1, 150, 50, step=1)

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        profile_pct = st.slider("Row percentile for band", 50, 100, 85, step=1)
    with c7:
        band_frac = st.slider("Band fraction of peak", 0.5, 1.0, 0.85, step=0.01)
    with c8:
        band_margin = st.slider("Band vertical margin (px)", 0, 100, 20, step=1)
    with c9:
        min_area_frac = st.slider("Min area fraction", 0.0, 0.5, 0.01, step=0.005)
    with c10:
        min_aspect = st.slider("Min aspect (w/h)", 1.0, 20.0, 5.0, step=0.5)

    alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="bb_alpha")

    heatmap, binary, overlay = method_bright_band(
        bgr=bgr,
        gray=gray,
        use_clahe=use_clahe,
        clahe_clip=clahe_clip,
        clahe_grid=clahe_grid,
        bilateral_d=bilateral_d,
        bilateral_sigma=bilateral_sigma,
        profile_pct=profile_pct,
        band_frac_of_peak=band_frac,
        band_margin=band_margin,
        min_area_frac=min_area_frac,
        min_aspect=min_aspect,
        alpha=alpha,
    )

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Intensity Heatmap (preprocessed)")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary Weld Region (filtered)")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.expander("Download results"):
        download_png_button(heatmap, "Download Heatmap", key="bb_heatmap")
        download_png_button(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Download Binary", key="bb_binary")
        download_png_button(overlay, "Download Overlay", key="bb_overlay")


st.markdown("---")
st.caption(
    "Tip: If your weld appears tilted or vertical, rotate the input before upload. This lab assumes the weld roughly spans horizontally."
)
