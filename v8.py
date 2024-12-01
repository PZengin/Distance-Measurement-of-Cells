import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, measure, morphology
from scipy.spatial import distance
from tifffile import TiffFile

# Define pixel size in nanometers (8.21 Å = 0.821 nm)
pixel_size_nm = 3.284

# Path to the folder containing TIFF files
input_folder = "/Users/pelinzengin/Desktop/Internship_Pos_lab"

# Path to the output CSV file
output_csv = "/Users/pelinzengin/Desktop/BW25II3_distances/im_om_distances2.csv"

# Ensure the output directory exists
output_dir = os.path.dirname(output_csv)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Morphological parameters
min_cell_area = 1000  # Minimum area of a cell to consider (in pixels)
closing_radius = 10  # Radius for morphological closing to connect fragments

# Initialize a list to store distances and their associated file names
distance_data = []

def plot_image_with_measurements(file_path, pixel_size_nm):
    """
    Plots a single image to visualize the segmentation and measurements.
    """
    # Load the TIFF file
    with TiffFile(file_path) as tif:
        tif_stack = tif.asarray()

    # Extract OM and IM slices
    outer_membrane = tif_stack[0]  # OM slice
    inner_membrane = tif_stack[1]  # IM slice

    # Apply thresholds to isolate membranes
    om_binary = outer_membrane > filters.threshold_otsu(outer_membrane)
    im_binary = inner_membrane > filters.threshold_otsu(inner_membrane)

    # Combine binary images and close small gaps
    combined_binary = om_binary | im_binary
    closed_binary = morphology.binary_closing(combined_binary, morphology.disk(closing_radius))

    # Label connected components
    labeled_cells, _ = measure.label(closed_binary, connectivity=2, return_num=True)

    # Filter cells by size
    cell_props = measure.regionprops(labeled_cells)
    valid_cells = [cell for cell in cell_props if cell.area >= min_cell_area]

    if not valid_cells:
        print("No valid cells found for plotting.")
        return

    # Plot one valid cell with measurements
    cell = valid_cells[0]  # Select the first valid cell
    cell_mask = labeled_cells == cell.label

    # Extract coordinates for OM and IM
    outer_coords = np.argwhere(cell_mask & om_binary)  # Outer membrane coordinates (OM)
    inner_coords = np.argwhere(cell_mask & im_binary)  # Inner membrane coordinates (IM)

    if outer_coords.size == 0 or inner_coords.size == 0:
        print("No valid OM or IM coordinates found for plotting.")
        return

    # Calculate pairwise distances between IM and OM
    pairwise_distances = distance.cdist(inner_coords, outer_coords)

    # Randomly sample 50 points on the inner membrane (IM)
    sampled_indices = np.random.choice(inner_coords.shape[0], min(50, inner_coords.shape[0]), replace=False)
    sampled_inner_coords = inner_coords[sampled_indices]

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cell_mask, cmap="gray", interpolation="none")

    # Add measurement lines (without highlighting OM/IM coordinates with dots)
    for idx in sampled_indices:
        closest_outer_idx = np.argmin(pairwise_distances[idx])
        plt.plot(
            [inner_coords[idx, 1], outer_coords[closest_outer_idx, 1]],
            [inner_coords[idx, 0], outer_coords[closest_outer_idx, 0]],
            c="yellow",  # Highlight the measurement line in yellow
            lw=1  # Increase line width to make the measurement lines more visible
        )

    plt.title("Cell with Membranes and Measurement Lines")
    plt.axis('off')  # Hide axis
    plt.show()


# Process each TIFF file in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith(".tif"):
        print(f"Processing file: {file_name}")
        file_path = os.path.join(input_folder, file_name)

        # Load the TIFF file
        try:
            with TiffFile(file_path) as tif:
                tif_stack = tif.asarray()
        except Exception as e:
            print(f"Error reading file {file_name}: {e}")
            continue

        # Ensure the stack has at least two slices
        if tif_stack.ndim != 3 or tif_stack.shape[0] < 2:
            print(f"Skipping file {file_name}: Invalid stack dimensions")
            continue

        # Extract the OM and IM slices
        outer_membrane = tif_stack[0]  # OM slice
        inner_membrane = tif_stack[1]  # IM slice

        # Apply a threshold to isolate membranes
        om_binary = outer_membrane > filters.threshold_otsu(outer_membrane)
        im_binary = inner_membrane > filters.threshold_otsu(inner_membrane)

        # Combine the two binary images
        combined_binary = om_binary | im_binary

        # Apply morphological closing to connect fragmented regions
        closed_binary = morphology.binary_closing(combined_binary, morphology.disk(closing_radius))

        # Label connected components
        labeled_cells, num_cells = measure.label(closed_binary, connectivity=2, return_num=True)

        # Filter out small regions (likely noise)
        cell_props = measure.regionprops(labeled_cells)
        valid_cells = [cell for cell in cell_props if cell.area >= min_cell_area]

        print(f"Found {len(valid_cells)} valid cells in {file_name}")
        if not valid_cells:
            continue

        # Process each valid cell
        for cell_id, cell in enumerate(valid_cells, start=1):
            # Create a mask for the current cell
            cell_mask = labeled_cells == cell.label

            # Extract the inner and outer membrane pixels for the cell
            outer_coords = np.argwhere(cell_mask & om_binary)  # Outer membrane coordinates (OM)
            inner_coords = np.argwhere(cell_mask & im_binary)  # Inner membrane coordinates (IM)

            if outer_coords.size == 0 or inner_coords.size == 0:
                print(f"Skipping cell {cell_id} in {file_name}: No membrane coordinates found.")
                continue

            # Calculate pairwise distances between inner and outer membrane pixels
            pairwise_distances = distance.cdist(inner_coords, outer_coords)

            # Measure the minimum distance at 50 random points on the inner membrane
            sampled_indices = np.random.choice(inner_coords.shape[0], min(50, inner_coords.shape[0]), replace=False)
            distances = [pairwise_distances[i].min() * pixel_size_nm for i in sampled_indices]

            # Store results with the file name and cell ID
            distance_data.extend([(file_name, cell_id, d) for d in distances])

# Create a DataFrame to save distances
df = pd.DataFrame(distance_data, columns=["File Name", "Cell ID", "Distance (nm)"])

# Check if data is collected
if df.empty:
    print("No data collected. Please check your image preprocessing and labeling.")
else:
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Data saved to {output_csv}")

# Example of visualizing a single image
example_file = "/Users/pelinzengin/Desktop/Internship_Pos_lab/images54_binned.tif"  # Update this path to your image
plot_image_with_measurements(example_file, pixel_size_nm)
