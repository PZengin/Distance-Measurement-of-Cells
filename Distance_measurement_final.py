import numpy as np
import matplotlib.pyplot as plt
from tifffile import TiffFile
import os
from skimage import filters, measure, morphology
import re
from skimage.morphology import binary_erosion, disk
import pandas as pd

# Define pixel size in nanometers (8.21 Å = 0.821 nm)
PIXEL_SIZE_NM = 32.84
SPACING_NM = 500  # Spacing between points in nm (updated to 500 nm)
MIN_CELL_AREA = 500  # Minimum area of a cell to consider (in pixels)
CLOSING_RADIUS = 10  # Radius for morphological closing to connect fragments


def load_tiff_stack(file_path):
    """Load a TIFF file and return the stack as a numpy array."""
    try:
        with TiffFile(file_path) as tif:
            return tif.asarray()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def preprocess_membranes(tif_stack, IO):
    """Preprocess TIFF stack and extract binary masks for outer and inner membranes."""
    if tif_stack.ndim != 3 or tif_stack.shape[0] < 2:
        return None

    if IO == 0:
        membrane = tif_stack[0]  # OM slice
    else:
        membrane = tif_stack[1]  # IM slice

    binary = membrane > filters.threshold_otsu(membrane)

    return binary


def extract_outermost_layer(binary_image, erosion_radius=5):
    """Extracts the outermost layer (boundary) of a binary structure."""
    eroded_image = binary_erosion(binary_image, footprint=disk(erosion_radius))
    outermost_layer = binary_image & ~eroded_image
    return outermost_layer


def extract_number(file_name):
    match = re.search(r'\d+', file_name)  # Look for numbers in the filename
    return int(match.group()) if match else float('inf')  # Use a high value for files without numbers


def get_distances(inner_coords, outer_coords):
    """Calculate distances between inner and outer membrane coordinates."""
    distances = []
    selected_inner_coords = []
    selected_outer_coords = []

    inner_perimeter = len(inner_coords)
    step_size = int(SPACING_NM / PIXEL_SIZE_NM)  # Spacing in pixels

    # Initial selection of inner points based on spacing
    initial_inner_coords = [inner_coords[i * step_size] for i in range(int(inner_perimeter // step_size))]

    # Greedily match the closest outer membrane points to the selected inner membrane points
    outer_coords_available = outer_coords.copy()
    for inner_point in initial_inner_coords:
        # Calculate the distances between the current inner point and all outer points
        distances_to_outer = np.linalg.norm(outer_coords_available - inner_point, axis=1)

        if distances_to_outer.size == 0:
            continue

        closest_outer_idx = np.argmin(distances_to_outer)

        # Get the closest outer point
        outer_point = outer_coords_available[closest_outer_idx]

        # Save the selected points for plotting
        selected_outer_coords.append(outer_point)

        # Remove the used outer point from the available list
        outer_coords_available = np.delete(outer_coords_available, closest_outer_idx, axis=0)

    # Reselect inner points to be the closest to the selected outer points
    inner_coords_array = np.array(inner_coords)
    for outer_point in selected_outer_coords:
        distances_to_inner = np.linalg.norm(inner_coords_array - outer_point, axis=1)
        closest_inner_idx = np.argmin(distances_to_inner)

        inner_point = inner_coords_array[closest_inner_idx]
        selected_inner_coords.append(inner_point)

        # Calculate and record the final distance
        dist = np.linalg.norm(inner_point - outer_point) * PIXEL_SIZE_NM
        distances.append(dist)

    return distances, selected_inner_coords, selected_outer_coords


import os


def plot_measurements(cell, distances, selected_inner_coords, selected_outer_coords, cell_id, file_name, save_dir=None):
    """Plot measurements of inner and outer membrane points with an option to save the plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cell, cmap='gray')

    for i in range(len(distances)):
        inner_pixel = selected_inner_coords[i]
        outer_pixel = selected_outer_coords[i]

        ax.plot([inner_pixel[1], outer_pixel[1]], [inner_pixel[0], outer_pixel[0]], 'r-', lw=1)
        ax.plot(inner_pixel[1], inner_pixel[0], 'bo', markersize=3)  # Inner membrane pixel (blue)
        ax.plot(outer_pixel[1], outer_pixel[0], 'go', markersize=3)  # Outer membrane pixel (green)

    ax.set_title(f'Cell {cell_id} - {file_name}')
    plt.axis('off')

    if save_dir:
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Save the plot as an image file
        save_path = os.path.join(save_dir, f'{cell_id}_{file_name}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def process_membranes(input_folder_IM, input_folder_OM, output_folder_plots=None):
    """Main processing loop for TIFF files."""
    distance_data = []

    for file_name in sorted(os.listdir(input_folder_IM), key=extract_number):
        if not file_name.endswith(".tif"):
            continue

        print(f"\rProcessing file: {file_name}", end="", flush=True)
        file_path_IM = os.path.join(input_folder_IM, file_name)
        file_path_OM = os.path.join(input_folder_OM, file_name)

        tif_stack_IM = load_tiff_stack(file_path_IM)
        if tif_stack_IM is None:
            continue

        tif_stack_OM = load_tiff_stack(file_path_OM)
        if tif_stack_OM is None:
            continue

        om_binary = preprocess_membranes(tif_stack_OM, 0)
        im_binary = preprocess_membranes(tif_stack_IM, 1)
        im_binary = extract_outermost_layer(im_binary)

        if om_binary is None or im_binary is None:
            print(f"Skipping file {file_name}: Invalid stack dimensions")
            continue

        outer_coords = np.argwhere(om_binary)
        inner_coords = np.argwhere(im_binary)

        if outer_coords.size == 0 or inner_coords.size == 0:
            print(f"Skipping file {file_name}: No membrane coordinates found.")
            continue

        distances, selected_inner, selected_outer = get_distances(inner_coords, outer_coords)

        mean_distance = np.mean(distances)

        # Define lower and upper boundaries
        lower_bound = 0.3 * mean_distance  # Adjust this factor as needed
        upper_bound = 3 * mean_distance

        # Filter indices based on both boundaries
        valid_indices = [i for i, d in enumerate(distances) if lower_bound <= d <= upper_bound]

        # Apply filtering to distances and corresponding lists
        filtered_distances = [distances[i] for i in valid_indices]
        filtered_selected_inner = [selected_inner[i] for i in valid_indices]
        filtered_selected_outer = [selected_outer[i] for i in valid_indices]

        distance_data.extend([(file_name, cell, d) for cell, d in enumerate(filtered_distances, 1)])

        plot_measurements(om_binary | im_binary, filtered_distances, filtered_selected_inner, filtered_selected_outer, cell_id=1, file_name=file_name, save_dir=output_folder_plots)

    return distance_data


# Define paths and run the script
input_folder_IM = "/Users/mfras/Downloads/Teddy_Code_Praktikum/Segmented_IM"
input_folder_OM = "/Users/mfras/Downloads/Teddy_Code_Praktikum/Segmented_OM"
output_csv = "/Users/mfras/Downloads/Teddy_Code_Praktikum/data.csv"
output_folder_plots = "/Users/mfras/Downloads/Teddy_Code_Praktikum/Plots"

distance_results = process_membranes(input_folder_IM, input_folder_OM, output_folder_plots)

df = pd.DataFrame(distance_results, columns=["File Name", "Cell ID", "Distance"])

if df.empty:
    print("No data collected. Please check your image preprocessing and labeling.")
else:
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print("---->")
    print(f"Data saved to {output_csv}")
