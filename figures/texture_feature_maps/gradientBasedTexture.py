import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter
# from mpl_toolkits.axes_grid1 import make_axes_locatable  # For aligned colorbars
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # For custom-sized colorbars

# --- Step 1: Load Grayscale Image ---
def load_image(image_path):
    image = io.imread(image_path, as_gray=True)
    return img_as_float(image)  # Ensure floating-point format for precision

def injectRandomNoise(image):
    image += np.random.normal(0, 1e-6, image.shape)
    return image

# --- Step 2: Compute Gradient Magnitude ---
def compute_gradient(image):
    grad_x, grad_y = np.gradient(image)
    return np.sqrt(grad_x**2 + grad_y**2)

# --- Step 3: Compute Statistical Maps (Variance, Skewness, Kurtosis) ---
def compute_statistical_maps(image, window_size=5, crop_padding=True):
    pad_width = window_size//2
    padded_image = np.pad(image, pad_width=pad_width, mode='reflect')
    # variance_map = np.zeros_like(image)
    # skewness_map = np.zeros_like(image)
    # kurtosis_map = np.zeros_like(image)
    variance_map = np.zeros_like(padded_image)
    skewness_map = np.zeros_like(padded_image)
    kurtosis_map = np.zeros_like(padded_image)

    # for i in range(image.shape[0]):
    #     for j in range(image.shape[1]):
    #         window = padded_img[i:i + window_size, j:j + window_size].flatten()
    #         variance_map[i, j] = np.var(window)
    #         skewness_map[i, j] = skew(window)
    #         kurtosis_map[i, j] = kurtosis(window, fisher=True)

    # Step 3: Compute statistics using sliding window
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded_image[i:i + window_size, j:j + window_size]

            variance_map[i + pad_width, j + pad_width] = np.var(window)
            skewness_map[i + pad_width, j + pad_width] = skew(window, axis=None)
            kurtosis_map[i + pad_width, j + pad_width] = kurtosis(window, axis=None)

    if crop_padding == True:
        variance_map = variance_map[pad_width:-pad_width, pad_width:-pad_width]
        skewness_map = skewness_map[pad_width:-pad_width, pad_width:-pad_width]
        kurtosis_map = kurtosis_map[pad_width:-pad_width, pad_width:-pad_width]

    # Optional: Smooth maps for better visualization
    return (
        gaussian_filter(variance_map, sigma=1),
        gaussian_filter(skewness_map, sigma=1),
        gaussian_filter(kurtosis_map, sigma=1)
    )

# --- Step 4: Visualize Results ---
def visualize_maps(image, variance_map, skewness_map, kurtosis_map, show_colorbar=False):
    plt.figure(figsize=(15, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image',fontweight='bold')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(variance_map, cmap='viridis')
    if show_colorbar==True:
        plt.colorbar(label='Gradient Variance')
    plt.title('Gradient Variance Map',fontweight='bold')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(skewness_map, cmap='plasma')
    if show_colorbar==True:
        plt.colorbar(label='Gradient Skewness')
    plt.title('Gradient Skewness Map',fontweight='bold')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(kurtosis_map, cmap='inferno')
    if show_colorbar==True:
        plt.colorbar(label='Gradient Kurtosis')
    plt.title('Gradient Kurtosis Map',fontweight='bold')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --- Step 5: Main Execution ---
image_path = "/home/josh/clotsimnet/data/cnn_data_test/cnn_data_crop/aN_447_rp_01700_seed_500_crop.jpeg" # Replace with image path
image = load_image(image_path)
print('Image loaded')
image = injectRandomNoise(image)
print('Noise injected')
gradient_magnitude = compute_gradient(image)
print('Gradient computed')
print('Computing statistical maps')
variance_map, skewness_map, kurtosis_map = compute_statistical_maps(gradient_magnitude, window_size=7)
print('Maps computed')
print('Visualizing maps')
visualize_maps(image, variance_map, skewness_map, kurtosis_map)
