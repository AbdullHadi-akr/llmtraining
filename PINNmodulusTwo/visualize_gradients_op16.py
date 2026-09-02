"""Visualize spatial gradients (dT/dx, dT/dy, dT/dz) at specific points for OP16.

Shows how temperature gradients evolve over time at representative points in:
- JR1 Center (heat source region)
- Cell Center (center of the battery)
- Gehäusewand (housing wall / boundary)
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = PROJECT_ROOT / "data_cache"
COORDS_DIR = PROJECT_ROOT / "coordinates"

# Load OP16 data
print("Loading OP16 data...")
op16_data = np.load(DATA_CACHE / "OP16.npz")
print(f"Available keys: {list(op16_data.keys())}")

# Load temperature field (T is shape (n_time, n_points))
temperature = op16_data["T"]  # (n_time, n_points)
time_values = op16_data["t_fast"]  # (n_time,)
coords = op16_data["xyz"]  # (n_points, 3) - x, y, z

print(f"Temperature shape: {temperature.shape}")
print(f"Time shape: {time_values.shape}")
print(f"Coords shape: {coords.shape}")
print(f"Time range: {time_values.min():.2f} to {time_values.max():.2f} seconds")

# Load coordinate reference files to identify regions
jr1_coords = pd.read_csv(COORDS_DIR / "Coordinates - Grid JR1 Center.csv")
cc_coords = pd.read_csv(COORDS_DIR / "Coordinates - Grid Cell Center.csv")
geh_coords = pd.read_csv(COORDS_DIR / "Coordinates - Grid Gehäusewand.csv")

print(f"\nJR1 Center: {len(jr1_coords)} points")
print(f"Cell Center: {len(cc_coords)} points")
print(f"Gehäusewand: {len(geh_coords)} points")


def find_closest_point(target_coords, all_coords):
    """Find index of closest point in all_coords to target_coords."""
    distances = np.sqrt(np.sum((all_coords - target_coords)**2, axis=1))
    return np.argmin(distances)


# Select representative points
# JR1 Center: middle point (heat source region)
jr1_mid_idx = len(jr1_coords) // 2
jr1_target = np.array([
    jr1_coords["Position[X] (m)"].iloc[jr1_mid_idx],
    jr1_coords["Position[Y] (m)"].iloc[jr1_mid_idx],
    jr1_coords["Position[Z] (m)"].iloc[jr1_mid_idx]
])

# Cell Center: middle point (x≈0)
cc_mid_idx = len(cc_coords) // 2
cc_target = np.array([
    cc_coords["Position[X] (m)"].iloc[cc_mid_idx],
    cc_coords["Position[Y] (m)"].iloc[cc_mid_idx],
    cc_coords["Position[Z] (m)"].iloc[cc_mid_idx]
])

# Gehäusewand: middle point (boundary)
geh_mid_idx = len(geh_coords) // 2
geh_target = np.array([
    geh_coords["Position[X] (m)"].iloc[geh_mid_idx],
    geh_coords["Position[Y] (m)"].iloc[geh_mid_idx],
    geh_coords["Position[Z] (m)"].iloc[geh_mid_idx]
])

# Find corresponding indices in the full grid
jr1_idx = find_closest_point(jr1_target, coords)
cc_idx = find_closest_point(cc_target, coords)
geh_idx = find_closest_point(geh_target, coords)

print(f"\nSelected points:")
print(f"JR1 Center point {jr1_idx}: {coords[jr1_idx]}")
print(f"Cell Center point {cc_idx}: {coords[cc_idx]}")
print(f"Gehäusewand point {geh_idx}: {coords[geh_idx]}")


def compute_spatial_gradients(temperature, coords, point_idx):
    """Compute spatial gradients dT/dx, dT/dy, dT/dz at a point using finite differences.
    
    Uses a simple nearest-neighbor approach to estimate gradients.
    """
    n_time = temperature.shape[0]
    gradients = np.zeros((n_time, 3))  # (time, [dx, dy, dz])
    
    # Find nearest neighbors in each direction
    point_coord = coords[point_idx]
    
    # Find neighbors by looking at nearby points
    distances = np.sqrt(np.sum((coords - point_coord)**2, axis=1))
    # Get indices of nearby points (excluding the point itself)
    nearby_mask = (distances > 0) & (distances < 0.02)  # within 2cm
    nearby_indices = np.where(nearby_mask)[0]
    
    if len(nearby_indices) < 3:
        print(f"Warning: Only {len(nearby_indices)} neighbors found for point {point_idx}")
        return gradients
    
    # For each timestep, compute gradients
    for t in range(n_time):
        T_point = temperature[t, point_idx]
        
        # Find neighbors in +x, +y, +z directions
        for dim in range(3):
            # Find points that are mainly displaced in this dimension
            delta = coords[nearby_indices] - point_coord
            dim_displacement = np.abs(delta[:, dim])
            other_dims = np.sqrt(np.sum(delta[:, [i for i in range(3) if i != dim]]**2, axis=1))
            
            # Points where displacement is mainly in this dimension
            valid = dim_displacement > 2 * other_dims
            if np.any(valid):
                valid_neighbors = nearby_indices[valid]
                # Use the closest valid neighbor
                neighbor_distances = distances[valid_neighbors]
                nearest_neighbor = valid_neighbors[np.argmin(neighbor_distances)]
                
                dx = coords[nearest_neighbor, dim] - point_coord[dim]
                dT = temperature[t, nearest_neighbor] - T_point
                if abs(dx) > 1e-6:
                    gradients[t, dim] = dT / dx
    
    return gradients


# Compute gradients at each point
print("\nComputing gradients...")
gradients_jr1 = compute_spatial_gradients(temperature, coords, jr1_idx)
gradients_cc = compute_spatial_gradients(temperature, coords, cc_idx)
gradients_geh = compute_spatial_gradients(temperature, coords, geh_idx)

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle("Spatial Temperature Gradients for OP16", fontsize=16, fontweight='bold')

time_minutes = time_values / 60.0  # Convert to minutes

locations = [
    ("JR1 Center (Heat Source)", gradients_jr1, coords[jr1_idx]),
    ("Cell Center", gradients_cc, coords[cc_idx]),
    ("Gehäusewand (Housing Wall)", gradients_geh, coords[geh_idx])
]

gradient_labels = ["dT/dx", "dT/dy", "dT/dz"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for row, (location_name, gradients, coord) in enumerate(locations):
    for col, (label, color) in enumerate(zip(gradient_labels, colors)):
        ax = axes[row, col]
        ax.plot(time_minutes, gradients[:, col], color=color, linewidth=2)
        ax.set_xlabel("Time (minutes)", fontsize=10)
        ax.set_ylabel(f"{label} (K/m)", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Title with location info
        if col == 0:
            title = f"{location_name}\n{label}"
        else:
            title = label
        ax.set_title(title, fontsize=11, fontweight='bold')
        
        # Add coordinate info to first plot of each row
        if col == 2:
            info_text = f"x={coord[0]:.4f}\ny={coord[1]:.4f}\nz={coord[2]:.4f}"
            ax.text(1.05, 0.5, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / "artifacts" / "op16_gradients_visualization.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

plt.show()

# Print summary statistics
print("\n" + "="*80)
print("GRADIENT SUMMARY STATISTICS")
print("="*80)

for location_name, gradients, coord in locations:
    print(f"\n{location_name}:")
    print(f"  Coordinates: x={coord[0]:.4f}, y={coord[1]:.4f}, z={coord[2]:.4f}")
    for i, label in enumerate(gradient_labels):
        grad_values = gradients[:, i]
        print(f"  {label}:")
        print(f"    Mean: {np.mean(grad_values):10.2f} K/m")
        print(f"    Std:  {np.std(grad_values):10.2f} K/m")
        print(f"    Min:  {np.min(grad_values):10.2f} K/m")
        print(f"    Max:  {np.max(grad_values):10.2f} K/m")
