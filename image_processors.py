import numpy as np
from PIL import Image
from kmeans_pp_nd import kmeans_plus_plus_init, kmeans_plus_plus_local_search_weighted, compute_kmeans_cost
from Exponential_quadtree_nd import exponential_quadtree_coreset


def load_image_as_rgb_array(image_path):
    """
    Load an image and convert it to an array of [x, y, r, g, b] for each pixel.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    rgb_array : np.ndarray of shape (n_pixels, 5)
        Each row is [x, y, r, g, b].
    image_shape : tuple
        Original image shape (height, width, channels).
    """
    img = Image.open(image_path)
    img_array = np.array(img)
    height, width, _ = img_array.shape

    # Create coordinate grid
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    # Flatten RGB values
    r = img_array[:, :, 0].flatten()
    g = img_array[:, :, 1].flatten()
    b = img_array[:, :, 2].flatten()

    rgb_array = np.column_stack([x_coords, y_coords, r, g, b])

    return rgb_array, (height, width, 3)


def compress_image_with_coreset(image_path, t, eps=0.1, random_state=None):
    """
    Compress an image using coreset-based k-means on RGB values.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    t : int
        Number of colors (centers) to use.
    eps : float
        Approximation parameter for coreset.
    random_state : int or None
        Random seed.

    Returns
    -------
    compressed_img : np.ndarray
        Reconstructed image array.
    original_shape : tuple
        Original image shape.
    stats : dict
        Dictionary containing:
        - full_size: number of original pixels
        - coreset_size: number of coreset points
        - initial_cost: cost with initial centers
        - final_cost: cost with final centers
    """
    # Load image as RGB array
    rgb_array, original_shape = load_image_as_rgb_array(image_path)
    height, width, _ = original_shape

    # Extract RGB points for clustering (ignore x,y for clustering)
    rgb_points = rgb_array[:, 2:5].astype(float)  # r, g, b

    # Initial k-means++ on RGB data
    initial_centers = kmeans_plus_plus_init(rgb_points, t, random_state=random_state)
    
    # Compute cost with initial centers
    initial_cost = compute_kmeans_cost(rgb_points, initial_centers)

    # Build coreset on RGB data
    coreset_points, coreset_weights, _ = exponential_quadtree_coreset(rgb_points, initial_centers, eps, random_state=random_state)

    # Weighted k-means++ local search on coreset
    final_centers, _ = kmeans_plus_plus_local_search_weighted(
        coreset_points,
        coreset_weights,
        t,
        n_steps=100,
        random_state=random_state
    )
    
    # Compute cost with final centers
    final_cost = compute_kmeans_cost(rgb_points, final_centers)

    # Assign each pixel to nearest center
    distances = np.linalg.norm(rgb_points[:, np.newaxis] - final_centers[np.newaxis, :], axis=2)
    nearest_center_indices = np.argmin(distances, axis=1)

    # Reconstruct image: each pixel gets color of its nearest center
    compressed_rgb = final_centers[nearest_center_indices].astype(np.uint8)

    # Reshape back to image
    compressed_img = compressed_rgb.reshape((height, width, 3))
    
    # Compile statistics
    stats = {
        'full_size': rgb_points.shape[0],
        'coreset_size': coreset_points.shape[0],
        'initial_cost': initial_cost,
        'final_cost': final_cost,
    }

    return compressed_img, original_shape, stats


def save_compressed_image(compressed_img, output_path):
    """
    Save the compressed image.

    Parameters
    ----------
    compressed_img : np.ndarray
        Compressed image array.
    output_path : str
        Path to save the image.
    """
    img = Image.fromarray(compressed_img)
    img.save(output_path)