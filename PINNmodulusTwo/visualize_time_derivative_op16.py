"""Visualize temporal derivative (dT/dt) at specific points for OP16.

Shows how the temperature change rate evolves over time at representative points in:
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


# Compute temporal derivatives dT/dt using finite differences
def compute_time_derivative(temperature, time_values, point_idx):
    """Compute dT/dt at a specific point using centered finite differences."""
    T_point = temperature[:, point_idx]
    n_time = len(time_values)
    dT_dt = np.zeros(n_time)
    
    # Forward difference for first point
    dT_dt[0] = (T_point[1] - T_point[0]) / (time_values[1] - time_values[0])
    
    # Centered difference for middle points
    for i in range(1, n_time - 1):
        dT_dt[i] = (T_point[i+1] - T_point[i-1]) / (time_values[i+1] - time_values[i-1])
    
    # Backward difference for last point
    dT_dt[-1] = (T_point[-1] - T_point[-2]) / (time_values[-1] - time_values[-2])
    
    return dT_dt, T_point


print("\nComputing temporal derivatives...")
dT_dt_jr1, T_jr1 = compute_time_derivative(temperature, time_values, jr1_idx)
dT_dt_cc, T_cc = compute_time_derivative(temperature, time_values, cc_idx)
dT_dt_geh, T_geh = compute_time_derivative(temperature, time_values, geh_idx)

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Temperature and Temporal Derivative (dT/dt) for OP16", fontsize=16, fontweight='bold')

time_minutes = time_values / 60.0  # Convert to minutes

locations = [
    ("JR1 Center (Heat Source)", T_jr1, dT_dt_jr1, coords[jr1_idx]),
    ("Cell Center", T_cc, dT_dt_cc, coords[cc_idx]),
    ("Gehäusewand (Housing Wall)", T_geh, dT_dt_geh, coords[geh_idx])
]

colors = ["#d62728", "#1f77b4", "#2ca02c"]

for row, (location_name, T_vals, dT_dt_vals, coord) in enumerate(locations):
    # Left column: Temperature
    ax_T = axes[row, 0]
    ax_T.plot(time_minutes, T_vals, color=colors[row], linewidth=2)
    ax_T.set_xlabel("Time (minutes)", fontsize=11)
    ax_T.set_ylabel("Temperature (K)", fontsize=11)
    ax_T.set_title(f"{location_name}\nTemperature", fontsize=12, fontweight='bold')
    ax_T.grid(True, alpha=0.3)
    
    # Add min/max annotation
    T_min, T_max = T_vals.min(), T_vals.max()
    ax_T.axhline(y=T_min, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_T.axhline(y=T_max, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_T.text(0.98, 0.05, f"ΔT = {T_max - T_min:.2f} K", 
             transform=ax_T.transAxes, ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Right column: dT/dt
    ax_dt = axes[row, 1]
    ax_dt.plot(time_minutes, dT_dt_vals, color=colors[row], linewidth=2)
    ax_dt.set_xlabel("Time (minutes)", fontsize=11)
    ax_dt.set_ylabel("dT/dt (K/s)", fontsize=11)
    ax_dt.set_title(f"{location_name}\nTemporal Derivative (dT/dt)", fontsize=12, fontweight='bold')
    ax_dt.grid(True, alpha=0.3)
    ax_dt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add coordinate info
    info_text = f"x={coord[0]:.4f} m\ny={coord[1]:.4f} m\nz={coord[2]:.4f} m"
    ax_dt.text(1.05, 0.5, info_text, transform=ax_dt.transAxes,
              fontsize=9, verticalalignment='center',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Highlight regions of interest in dT/dt
    # Find where heating rate is maximum
    max_idx = np.argmax(dT_dt_vals)
    ax_dt.plot(time_minutes[max_idx], dT_dt_vals[max_idx], 'ro', markersize=8)
    ax_dt.annotate(f'Max: {dT_dt_vals[max_idx]:.4f} K/s',
                  xy=(time_minutes[max_idx], dT_dt_vals[max_idx]),
                  xytext=(10, 10), textcoords='offset points',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                  arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()

# Save figure
output_path = Path(__file__).parent / "artifacts" / "op16_time_derivative_visualization.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization to: {output_path}")

plt.show()

# Print summary statistics
print("\n" + "="*80)
print("TEMPORAL DERIVATIVE SUMMARY STATISTICS")
print("="*80)

for location_name, T_vals, dT_dt_vals, coord in locations:
    print(f"\n{location_name}:")
    print(f"  Coordinates: x={coord[0]:.4f}, y={coord[1]:.4f}, z={coord[2]:.4f}")
    print(f"\n  Temperature:")
    print(f"    Initial: {T_vals[0]:10.2f} K")
    print(f"    Final:   {T_vals[-1]:10.2f} K")
    print(f"    Max:     {T_vals.max():10.2f} K (at t={time_minutes[np.argmax(T_vals)]:.2f} min)")
    print(f"    ΔT:      {T_vals[-1] - T_vals[0]:10.2f} K")
    print(f"\n  dT/dt:")
    print(f"    Mean:    {np.mean(dT_dt_vals):10.6f} K/s")
    print(f"    Std:     {np.std(dT_dt_vals):10.6f} K/s")
    print(f"    Max:     {np.max(dT_dt_vals):10.6f} K/s (at t={time_minutes[np.argmax(dT_dt_vals)]:.2f} min)")
    print(f"    Min:     {np.min(dT_dt_vals):10.6f} K/s (at t={time_minutes[np.argmin(dT_dt_vals)]:.2f} min)")

print("\n" + "="*80)
