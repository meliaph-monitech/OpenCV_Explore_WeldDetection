import io
from dataclasses import dataclass
from typing import Optional, Tuple, List

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


def heatmap_overlay(heatmap_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    # Guarantee heatmap is 3-channel BGR
    if heatmap_bgr.ndim == 2:
        base = cv2.cvtColor(heatmap_bgr, cv2.COLOR_GRAY2BGR)
    else:
        base = heatmap_bgr.copy()
    return overlay_mask_on_image(base, mask, alpha=alpha, color=(0, 0, 255))


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
# Optional global preprocessing blocks
# ---------------------------------------------

def preprocess_retinex(gray: np.ndarray, sigma: int = 15) -> np.ndarray:
    # Single-Scale Retinex (SSR)
    gray_f = gray.astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(gray_f, (oddify(sigma), oddify(sigma)), 0)
    log_im = np.log(gray_f)
    log_blur = np.log(blur + 1e-6)
    out = log_im - log_blur
    out = normalize01(out)
    return (out * 255).astype(np.uint8)


def preprocess_homomorphic(gray: np.ndarray, cutoff: float = 0.015, order: int = 2) -> np.ndarray:
    # Simple homomorphic filtering in frequency domain
    g = gray.astype(np.float32) / 255.0
    g = np.log1p(g)
    H, W = g.shape
    y, x = np.indices((H, W))
    cy, cx = H // 2, W // 2
    D2 = (y - cy) ** 2 + (x - cx) ** 2
    D0 = (max(H, W) * cutoff) ** 2
    # High-pass Butterworth
    Hf = 1.0 - 1.0 / (1.0 + (D2 / (D0 + 1e-6)) ** order)
    G = np.fft.fftshift(np.fft.fft2(g))
    S = Hf * G
    s = np.real(np.fft.ifft2(np.fft.ifftshift(S)))
    s = np.expm1(s)
    s = normalize01(s)
    return (s * 255).astype(np.uint8)

# ---------------------------------------------
# Existing Methods (1–4) + Sobel Band Edges (5)
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


# 2) Canny + fill largest region + dilation

def fill_largest_from_edges(edges: np.ndarray) -> np.ndarray:
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
    x = (mask1d.astype(np.float32) > 0).astype(np.uint8)
    xpad = np.r_[0, x, 0]
    d = np.diff(xpad)
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    if starts.size == 0 or ends.size == 0:
        return None
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
    q = np.percentile(g, profile_pct, axis=1).astype(np.float32)
    q_s = cv2.GaussianBlur(q.reshape(-1, 1), (1, 21), 0).ravel()
    peak = float(q_s.max()) + 1e-6
    band_mask_1d = q_s >= band_frac_of_peak * peak

    run = longest_run(band_mask_1d)
    if run is None:
        roi_mask = np.ones_like(g, np.uint8) * 255
    else:
        y0, y1 = run
        y0 = max(0, y0 - band_margin)
        y1 = min(H - 1, y1 + band_margin)
        roi_mask = np.zeros_like(g, np.uint8)
        roi_mask[y0:y1 + 1, :] = 255

    g_roi = cv2.bitwise_and(g, g, mask=roi_mask)
    _t, _ = cv2.threshold(g_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_img = cv2.threshold(g, int(_t), 255, cv2.THRESH_BINARY)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=roi_mask)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filt = np.zeros_like(bin_img)
    best_area = 0
    H, W = g.shape
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


# 5) Sobel Band Edges (upper/lower) – provided by user in their extension (placeholder)

def method_sobel_band(gray: np.ndarray,
                      sobel_ksize: int,
                      band_width: int,
                      thresh: int,
                      alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = 3 if sobel_ksize not in [1, 3, 5, 7] else sobel_ksize
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=k)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=k)
    mag = cv2.magnitude(gx, gy)
    mag8 = np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)
    _, edges = cv2.threshold(mag8, thresh, 255, cv2.THRESH_BINARY)
    # simple band by vertical distance transform (demo)
    ys, xs = np.where(edges > 0)
    mask = np.zeros_like(gray, np.uint8)
    if xs.size > 0:
        for x in range(gray.shape[1]):
            ycol = ys[xs == x]
            if ycol.size >= 1:
                y0, y1 = ycol.min(), ycol.max()
                y0 = max(0, y0 - band_width)
                y1 = min(gray.shape[0] - 1, y1 + band_width)
                mask[y0:y1 + 1, x] = 255
    heatmap = apply_heatmap01(normalize01(mag))
    overlay = heatmap_overlay(heatmap, mask, alpha)
    return heatmap, mask, overlay

# ---------------------------------------------
# New Methods (6–10)
# ---------------------------------------------

# 6) Sauvola local thresholding

def sauvola_threshold(gray: np.ndarray, window: int = 21, k: float = 0.2, R: float = 128.0) -> np.ndarray:
    w = oddify(max(3, int(window)))
    mean = cv2.boxFilter(gray.astype(np.float32), ddepth=-1, ksize=(w, w), normalize=True)
    mean_sq = cv2.boxFilter((gray.astype(np.float32) ** 2), ddepth=-1, ksize=(w, w), normalize=True)
    var = np.clip(mean_sq - mean ** 2, 0, None)
    std = np.sqrt(var)
    thresh = mean * (1 + k * (std / (R + 1e-6) - 1))
    bin_img = (gray.astype(np.float32) > thresh).astype(np.uint8) * 255
    return bin_img


def method_sauvola(gray: np.ndarray, window: int, k: float, post_open: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bin_img = sauvola_threshold(gray, window=window, k=k)
    if post_open > 0:
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, bin_kernel(post_open))
    heatmap = apply_heatmap01(normalize01(gray))
    overlay = heatmap_overlay(heatmap, bin_img, alpha)
    return heatmap, bin_img, overlay

# 7) Morphological top-hat pipeline

def method_tophat(gray: np.ndarray, ksize: int, se_shape: str, thresh: int, post_close: int, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k = bin_kernel(ksize)
    if se_shape == 'ellipse':
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    elif se_shape == 'rect':
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    # white tophat enhances bright stripes
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    _, bin_img = cv2.threshold(th, thresh, 255, cv2.THRESH_BINARY)
    if post_close > 0:
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, bin_kernel(post_close))
    heatmap = apply_heatmap01(normalize01(th))
    overlay = heatmap_overlay(heatmap, bin_img, alpha)
    return heatmap, bin_img, overlay

# 8) Gabor bank (orientation selective)

def gabor_bank_response(gray: np.ndarray, thetas: List[float], lambd: float, gamma: float, sigma: float) -> np.ndarray:
    accum = np.zeros_like(gray, dtype=np.float32)
    for theta in thetas:
        ksize = oddify(int(6 * sigma))
        kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi=0, ktype=cv2.CV_32F)
        resp = cv2.filter2D(gray.astype(np.float32), cv2.CV_32F, kern)
        accum = np.maximum(accum, resp)
    accum = normalize01(accum)
    return (accum * 255).astype(np.uint8)


def method_gabor(gray: np.ndarray, n_orients: int, lambd: float, gamma: float, sigma: float, thresh: int, post_close: int, alpha: float):
    thetas = [i * np.pi / n_orients for i in range(n_orients)]
    resp = gabor_bank_response(gray, thetas, lambd=lambd, gamma=gamma, sigma=sigma)
    _, bin_img = cv2.threshold(resp, thresh, 255, cv2.THRESH_BINARY)
    if post_close > 0:
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, bin_kernel(post_close))
    heatmap = apply_heatmap01(resp.astype(np.float32) / 255.0)
    overlay = heatmap_overlay(heatmap, bin_img, alpha)
    return heatmap, bin_img, overlay

# 9) Hough-guided ROI + local threshold

def method_hough_roi(bgr: np.ndarray, gray: np.ndarray, canny_low: int, ratio: float, hough_thresh: int, min_len: int, max_gap: int, roi_halfwidth: int, adapt_block: int, adapt_c: int, alpha: float):
    edges = cv2.Canny(gray, canny_low, int(canny_low * ratio))
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=hough_thresh, minLineLength=min_len, maxLineGap=max_gap)
    mask = np.zeros_like(gray, np.uint8)
    if lines is not None and len(lines) > 0:
        # choose longest line as the seam axis
        lens = [np.hypot(x2-x1, y2-y1) for [[x1,y1,x2,y2]] in lines]
        L = int(np.argmax(lens))
        x1, y1, x2, y2 = lines[L][0]
        # unit normal vector to the line
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy) + 1e-6
        nx, ny = -dy / norm, dx / norm
        H, W = gray.shape
        # draw a band around the line
        for t in np.linspace(0, 1, num=max(W, H)):
            cx = int(round(x1 + t * dx))
            cy = int(round(y1 + t * dy))
            if 0 <= cx < W and 0 <= cy < H:
                px = int(round(cx + nx * roi_halfwidth))
                py = int(round(cy + ny * roi_halfwidth))
                qx = int(round(cx - nx * roi_halfwidth))
                qy = int(round(cy - ny * roi_halfwidth))
                cv2.line(mask, (px, py), (qx, qy), 255, thickness=1)
        mask = cv2.dilate(mask, bin_kernel(roi_halfwidth*2+1), iterations=1)
    else:
        # fallback: whole image
        mask[:] = 255

    am = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    bin_img = cv2.adaptiveThreshold(gray, 255, am, cv2.THRESH_BINARY, oddify(adapt_block), adapt_c)
    bin_img = cv2.bitwise_and(bin_img, bin_img, mask=mask)
    heatmap = apply_heatmap01(normalize01(gray))
    overlay = overlay_mask_on_image(bgr, bin_img, alpha)
    return heatmap, bin_img, overlay

# 10) MSER detector (bright or dark)

def method_mser(bgr: np.ndarray, gray: np.ndarray, delta: int, min_area: int, max_area: int, max_var: float, bright: bool, alpha: float):
    # Some OpenCV builds don't accept keyword args in MSER_create; use defaults + setters for maximum compatibility
    try:
        mser = cv2.MSER_create()
    except Exception:
        # Fallback namespace variant seen in some builds
        mser = cv2.MSER_create()
    # Defensive casting
    mser.setDelta(int(delta))
    mser.setMinArea(int(min_area))
    mser.setMaxArea(int(max_area))
    try:
        mser.setMaxVariation(float(max_var))
    except Exception:
        # Some builds name it "setMaxVariation"; keep try just in case
        pass

    g = gray
    if not bright:
        g = 255 - g
    regions, _ = mser.detectRegions(g)

    mask = np.zeros_like(gray, np.uint8)
    for pts in regions:
        # Ensure int32 and proper shape for fillPoly
        poly = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [poly], 255)

    heatmap = apply_heatmap01(normalize01(gray))
    overlay = overlay_mask_on_image(bgr, mask, alpha)
    return heatmap, mask, overlay

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

bgr_raw = load_image_to_bgr(uploaded)
rgb = to_rgb_cached(bgr_raw)
gray_raw = to_gray_cached(bgr_raw)

# Optional global preprocessing (applies to gray for all methods)
st.sidebar.header("Global Preprocessing (optional)")
use_retinex = st.sidebar.toggle("Retinex (SSR)", value=False, key="sb_retinex")
retinex_sigma = st.sidebar.slider("Retinex σ", 5, 51, 15, step=2, key="sb_retinex_sigma")
use_homomorphic = st.sidebar.toggle("Homomorphic filter", value=False, key="sb_homo")
homo_cutoff = st.sidebar.slider("Homomorphic cutoff", 0.005, 0.05, 0.015, step=0.001, key="sb_homo_cutoff")
homo_order = st.sidebar.slider("Homomorphic order", 1, 4, 2, key="sb_homo_order")

gray = gray_raw.copy()
if use_retinex:
    gray = preprocess_retinex(gray, sigma=retinex_sigma)
if use_homomorphic:
    gray = preprocess_homomorphic(gray, cutoff=homo_cutoff, order=homo_order)

st.subheader("Preview")
st.image(to_rgb_cached(bgr_raw), caption=f"Input ({bgr_raw.shape[1]}×{bgr_raw.shape[0]})", use_column_width=True)

# Tabs for different methods
TAB_TITLES = [
    "Threshold (Intensity/Gradient)",
    "Canny + Fill Largest",
    "CLAHE + Otsu / Adaptive",
    "Bright-Band Guided",
    "Sobel Band Edges",
    "Sauvola Local",
    "Top-hat",
    "Gabor Bank",
    "Hough-ROI",
    "MSER"
]

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
    tab9,
    tab10,
) = st.tabs(TAB_TITLES)

with tab1:
    st.markdown("### 1) Threshold (Intensity or Gradient)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        use_gradient = st.toggle("Use gradient magnitude", value=False, key="t1_usegrad")
    with c2:
        threshold = st.slider("Threshold", min_value=0, max_value=255, value=140, step=1, key="t1_thresh")
    with c3:
        blur_ksize = st.slider("Gaussian blur ksize", 0, 21, 5, step=1, key="t1_blur")
    with c4:
        close_ksize = st.slider("Morph close ksize", 0, 21, 5, step=1, key="t1_close")
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="t1_alpha")

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

with tab2:
    st.markdown("### 2) Canny + Fill Largest Region")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        t_low = st.slider("Canny low", 0, 255, 60, step=1, key="t2_low")
    with c2:
        ratio = st.select_slider("High/Low ratio", options=[2.0, 2.5, 3.0, 3.5], value=3.0, key="t2_ratio")
        t_high = int(t_low * float(ratio))
        t_high = max(t_low + 1, min(255, t_high))
        st.write(f"High ≈ {t_high}")
    with c3:
        blur_ksize = st.slider("Pre-blur ksize", 0, 21, 5, step=1, key="t2_blur")
    with c4:
        dilate_iters = st.slider("Dilate iterations", 0, 5, 2, key="t2_dilate")
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="alpha_canny")

    heatmap, filled, overlay = method_canny_fill(bgr_raw, gray, t_low, t_high, blur_ksize, dilate_iters, alpha)

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

with tab3:
    st.markdown("### 3) CLAHE + Otsu / Adaptive Thresholding")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        use_clahe = st.toggle("Enable CLAHE", value=True, key="t3_clahe")
    with c2:
        clahe_clip = st.slider("CLAHE clipLimit", 0.1, 5.0, 2.0, step=0.1, key="t3_clip")
    with c3:
        clahe_grid = st.slider("CLAHE tileGridSize", 4, 32, 8, step=1, key="t3_grid")
    with c4:
        blur_ksize = st.slider("Gaussian blur ksize", 0, 21, 5, step=1, key="t3_blur")
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="t3_alpha")

    c6, c7, c8, c9 = st.columns(4)
    with c6:
        mode = st.radio("Threshold mode", ["Otsu", "Adaptive"], horizontal=True, key="t3_mode")
    with c7:
        otsu_bias = st.slider("Otsu bias (× T)", 0.5, 1.5, 1.0, step=0.05, key="t3_otsu")
    with c8:
        adapt_method = st.selectbox("Adaptive method", ["Mean", "Gaussian"], index=1, key="t3_am")
    with c9:
        adapt_block = st.slider("Adaptive block size (odd)", 3, 51, 21, step=2, key="t3_ab")
    c10, c11 = st.columns(2)
    with c10:
        adapt_c = st.slider("Adaptive C (stricter when higher)", -10, 20, 2, step=1, key="t3_ac")
    with c11:
        morph_ksize = st.slider("Morph open ksize", 0, 21, 5, step=1, key="t3_mk")
    morph_iters = st.slider("Morph open iters", 0, 5, 1, key="t3_mi")

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

with tab4:
    st.markdown("### 4) Bright-Band Guided Detection")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        use_clahe = st.toggle("Enable CLAHE", value=True, key="t4_clahe")
    with c2:
        clahe_clip = st.slider("CLAHE clipLimit", 0.1, 5.0, 2.0, step=0.1, key="t4_clip")
    with c3:
        clahe_grid = st.slider("CLAHE tileGridSize", 4, 32, 8, step=1, key="t4_grid")
    with c4:
        bilateral_d = st.slider("Bilateral d", 0, 15, 5, step=1, key="t4_bd")
    with c5:
        bilateral_sigma = st.slider("Bilateral sigma (color/space)", 1, 150, 50, step=1, key="t4_bs")

    c6, c7, c8, c9, c10 = st.columns(5)
    with c6:
        profile_pct = st.slider("Row percentile for band", 50, 100, 85, step=1, key="t4_pp")
    with c7:
        band_frac = st.slider("Band fraction of peak", 0.5, 1.0, 0.85, step=0.01, key="t4_bf")
    with c8:
        band_margin = st.slider("Band vertical margin (px)", 0, 100, 20, step=1, key="t4_bm")
    with c9:
        min_area_frac = st.slider("Min area fraction", 0.0, 0.5, 0.01, step=0.005, key="t4_maf")
    with c10:
        min_aspect = st.slider("Min aspect (w/h)", 1.0, 20.0, 5.0, step=0.5, key="t4_masp")

    alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="bb_alpha")

    heatmap, binary, overlay = method_bright_band(
        bgr=bgr_raw,
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

with tab5:
    st.markdown("### 5) Sobel Band Edges (Upper/Lower)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sobel_ksize = st.select_slider("Sobel ksize", options=[1,3,5,7], value=3, key="t5_sk")
    with c2:
        band_width = st.slider("Band half-width (px)", 0, 50, 10, key="t5_bw")
    with c3:
        thresh = st.slider("Edge threshold", 0, 255, 80, key="t5_th")
    with c4:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="sobel_alpha")

    heatmap, binary, overlay = method_sobel_band(gray, sobel_ksize, band_width, thresh, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Gradient Magnitude Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary Band")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab6:
    st.markdown("### 6) Sauvola Local Thresholding")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        window = st.slider("Window (odd)", 3, 101, 31, step=2, key="t6_win")
    with c2:
        k = st.slider("k (sensitivity)", 0.05, 0.8, 0.2, step=0.05, key="t6_k")
    with c3:
        post_open = st.slider("Morph open ksize", 0, 21, 5, step=1, key="t6_po")
    with c4:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="sauv_alpha")

    heatmap, binary, overlay = method_sauvola(gray, window, k, post_open, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Original/Preprocessed Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab7:
    st.markdown("### 7) Morphological Top-hat")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        ksize = st.slider("SE size", 3, 101, 31, step=2, key="t7_se")
    with c2:
        se_shape = st.selectbox("SE shape", ["ellipse", "rect", "ones"], index=0, key="t7_shape")
    with c3:
        thresh = st.slider("Threshold", 0, 255, 30, key="t7_th")
    with c4:
        post_close = st.slider("Morph close ksize", 0, 21, 5, step=1, key="t7_pc")
    with c5:
        alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="th_alpha")

    heatmap, binary, overlay = method_tophat(gray, ksize, se_shape, thresh, post_close, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Top-hat Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab8:
    st.markdown("### 8) Gabor Filter Bank")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_orients = st.select_slider("#Orientations", options=[4,6,8,12], value=8, key="t8_no")
    with c2:
        lambd = st.slider("Wavelength (λ)", 2.0, 40.0, 12.0, step=1.0, key="t8_la")
    with c3:
        gamma = st.slider("Gamma (aspect)", 0.1, 1.0, 0.5, step=0.05, key="t8_ga")
    with c4:
        sigma = st.slider("Sigma", 1.0, 20.0, 4.0, step=0.5, key="t8_si")
    with c5:
        thresh = st.slider("Threshold", 0, 255, 50, key="t8_th")
    post_close = st.slider("Morph close ksize", 0, 21, 5, step=1, key="t8_pc")
    alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="gabor_alpha")

    heatmap, binary, overlay = method_gabor(gray, n_orients, lambd, gamma, sigma, thresh, post_close, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Max Gabor Response Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab9:
    st.markdown("### 9) Hough-guided ROI + Local Threshold")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        canny_low = st.slider("Canny low", 0, 255, 60, step=1, key="t9_cl")
    with c2:
        ratio = st.select_slider("High/Low ratio", options=[2.0, 2.5, 3.0], value=3.0, key="t9_ratio")
    with c3:
        hough_thresh = st.slider("Hough threshold", 10, 200, 80, key="t9_ht")
    with c4:
        min_len = st.slider("Min line length", 10, 1000, 200, key="t9_ml")
    with c5:
        max_gap = st.slider("Max gap", 0, 200, 20, key="t9_mg")

    roi_halfwidth = st.slider("ROI half-width (px)", 1, 80, 20, key="t9_rhw")
    adapt_block = st.slider("Adaptive block size (odd)", 3, 101, 31, step=2, key="t9_ab")
    adapt_c = st.slider("Adaptive C", -20, 20, 2, step=1, key="t9_ac")
    alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="hough_alpha")

    heatmap, binary, overlay = method_hough_roi(bgr_raw, gray, canny_low, ratio, hough_thresh, min_len, max_gap, roi_halfwidth, adapt_block, adapt_c, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Edges / Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("Binary in ROI")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

with tab10:
    st.markdown("### 10) MSER Regions")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        delta = st.slider("Δ (step)", 1, 10, 5, key="t10_delta")
    with c2:
        min_area = st.slider("Min area", 30, 20000, 100, key="t10_min")
    with c3:
        max_area = st.slider("Max area", 1000, 200000, 50000, step=1000, key="t10_max")
    with c4:
        max_var = st.slider("Max variation", 0.1, 1.0, 0.5, step=0.05, key="t10_var")
    with c5:
        bright = st.toggle("Detect bright regions", value=True, key="t10_bright")
    alpha = st.slider("Overlay alpha", 0.0, 1.0, 0.5, step=0.05, key="mser_alpha")

    heatmap, binary, overlay = method_mser(bgr_raw, gray, delta, min_area, max_area, max_var, bright, alpha)

    cA, cB, cC = st.columns(3)
    with cA:
        st.caption("Base Heatmap")
        st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_column_width=True)
    with cB:
        st.caption("MSER Mask")
        st.image(binary, use_column_width=True)
    with cC:
        st.caption("Overlay")
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_column_width=True)

st.markdown("---")
st.caption(
    "Tip: If your weld appears tilted or vertical, rotate the input before upload. Global pre-processing applies to all tabs."
)
