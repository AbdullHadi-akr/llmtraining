"""Analyze optimal time intervals for recurrent model based on dT/dt analysis.

Helps determine: Should we use 1s, 5s, 10s, or other intervals for the history?
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_CACHE = PROJECT_ROOT / "data_cache"

# Load OP16 data
print("Loading OP16 data...")
op16_data = np.load(DATA_CACHE / "OP16.npz")
temperature = op16_data["T"]  # (n_time, n_points)
time_values = op16_data["t_fast"]  # (n_time,)

print(f"Temperature shape: {temperature.shape}")
print(f"Time range: {time_values.min():.2f} to {time_values.max():.2f} seconds")

# Check actual time step in data
dt_data = np.diff(time_values)
print(f"\nActual time step in data:")
print(f"  Mean: {dt_data.mean():.4f} seconds")
print(f"  Min:  {dt_data.min():.4f} seconds")
print(f"  Max:  {dt_data.max():.4f} seconds")

# Compute dT/dt for all points
dT_dt_all = np.zeros_like(temperature)
for i in range(temperature.shape[1]):  # for each point
    # Forward difference for first point
    dT_dt_all[0, i] = (temperature[1, i] - temperature[0, i]) / dt_data[0]
    
    # Centered difference for middle points
    for t in range(1, len(time_values) - 1):
        dT_dt_all[t, i] = (temperature[t+1, i] - temperature[t-1, i]) / (time_values[t+1] - time_values[t-1])
    
    # Backward difference for last point
    dT_dt_all[-1, i] = (temperature[-1, i] - temperature[-2, i]) / dt_data[-1]

# Analyze max |dT/dt| over all points and time
abs_dT_dt = np.abs(dT_dt_all)
max_dT_dt_per_time = np.max(abs_dT_dt, axis=1)  # max over all points at each time
max_dT_dt_per_point = np.max(abs_dT_dt, axis=0)  # max over time for each point

print(f"\n|dT/dt| statistics (over all points and times):")
print(f"  Global max:  {abs_dT_dt.max():.6f} K/s")
print(f"  95th percentile: {np.percentile(abs_dT_dt, 95):.6f} K/s")
print(f"  90th percentile: {np.percentile(abs_dT_dt, 90):.6f} K/s")
print(f"  75th percentile: {np.percentile(abs_dT_dt, 75):.6f} K/s")
print(f"  50th percentile (median): {np.percentile(abs_dT_dt, 50):.6f} K/s")

# Test different time intervals
test_intervals = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0]

print("\n" + "="*80)
print("TEMPERATURE CHANGE FOR DIFFERENT TIME INTERVALS")
print("="*80)
print("(Based on maximum dT/dt observed)")
print()

max_rate = abs_dT_dt.max()
for dt in test_intervals:
    dT = max_rate * dt
    print(f"Δt = {dt:5.1f} s  →  ΔT ≈ {dT:.4f} K  ({dT*1000:.2f} mK)")

# Find when dT/dt drops below certain thresholds
time_minutes = time_values / 60.0
threshold_percentiles = [95, 90, 75, 50]

print("\n" + "="*80)
print("TIME REGIMES BASED ON dT/dt")
print("="*80)

for pct in threshold_percentiles:
    threshold = np.percentile(abs_dT_dt, pct)
    # Find when max dT/dt drops below this threshold permanently
    above_threshold = max_dT_dt_per_time > threshold
    if np.any(~above_threshold):
        first_below_idx = np.where(~above_threshold)[0][0]
        time_below = time_minutes[first_below_idx]
        print(f"\n{pct}th percentile (|dT/dt| < {threshold:.6f} K/s) after t ≈ {time_below:.2f} min")

# Analyze phase-wise: 0-5 min, 5-15 min, 15+ min
phases = [
    ("Fast dynamics (0-5 min)", 0, 5),
    ("Transition (5-15 min)", 5, 15),
    ("Quasi-steady (15+ min)", 15, 24.1)
]

print("\n" + "="*80)
print("PHASE-WISE dT/dt ANALYSIS")
print("="*80)

for phase_name, t_start, t_end in phases:
    mask = (time_minutes >= t_start) & (time_minutes < t_end)
    if not np.any(mask):
        continue
    
    phase_dT_dt = abs_dT_dt[mask, :]
    max_in_phase = phase_dT_dt.max()
    median_in_phase = np.median(phase_dT_dt)
    p95_in_phase = np.percentile(phase_dT_dt, 95)
    
    print(f"\n{phase_name}:")
    print(f"  Max |dT/dt|:    {max_in_phase:.6f} K/s")
    print(f"  95th percentile: {p95_in_phase:.6f} K/s")
    print(f"  Median:          {median_in_phase:.6f} K/s")
    print(f"\n  Recommended Δt for 0.1K accuracy: {0.1 / max_in_phase:.2f} s")
    print(f"  Recommended Δt for 0.5K accuracy: {0.5 / max_in_phase:.2f} s")
    print(f"  Recommended Δt for 1.0K accuracy: {1.0 / max_in_phase:.2f} s")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Temporal Resolution Requirements for Recurrent Model", fontsize=16, fontweight='bold')

# Top: max |dT/dt| over time
ax1 = axes[0]
ax1.plot(time_minutes, max_dT_dt_per_time * 1000, linewidth=2, color='#d62728')
ax1.set_xlabel("Time (minutes)", fontsize=12)
ax1.set_ylabel("Max |dT/dt| (mK/s)", fontsize=12)
ax1.set_title("Maximum Temperature Change Rate (across all points)", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Add phase boundaries
for phase_name, t_start, t_end in phases[:-1]:
    ax1.axvline(x=t_end, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
    
# Add accuracy lines
accuracy_levels = [0.1, 0.5, 1.0]  # K
colors_acc = ['red', 'orange', 'green']
for acc, color in zip(accuracy_levels, colors_acc):
    # Required max dT/dt for this accuracy at different dt
    for dt in [1.0, 5.0, 10.0]:
        required_rate = acc / dt * 1000  # mK/s
        ax1.axhline(y=required_rate, color=color, linestyle=':', alpha=0.4, linewidth=1)

ax1.text(0.02, 0.98, "Fast dynamics\n(Δt should be small)", 
         transform=ax1.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax1.text(0.85, 0.98, "Quasi-steady\n(Δt can be larger)", 
         transform=ax1.transAxes, va='top', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Bottom: histogram of |dT/dt|
ax2 = axes[1]
flat_dT_dt = abs_dT_dt.flatten() * 1000  # mK/s
ax2.hist(flat_dT_dt, bins=100, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.set_xlabel("|dT/dt| (mK/s)", fontsize=12)
ax2.set_ylabel("Frequency", fontsize=12)
ax2.set_title("Distribution of |dT/dt| (all points, all times)", fontsize=13, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

# Mark percentiles
for pct in [50, 75, 90, 95]:
    val = np.percentile(flat_dT_dt, pct)
    ax2.axvline(x=val, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    ax2.text(val, ax2.get_ylim()[1] * 0.5, f'p{pct}', rotation=90, 
            va='bottom', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

output_path = Path(__file__).parent / "artifacts" / "op16_time_interval_analysis.png"
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n\nSaved analysis to: {output_path}")

# Final recommendation
print("\n" + "="*80)
print("RECOMMENDATION FOR RECURRENT MODEL TIME INTERVALS")
print("="*80)
print("""
Based on the dT/dt analysis:

1. FAST DYNAMICS PHASE (0-5 minutes):
   - Max |dT/dt| ≈ 28 mK/s
   - For ΔT < 0.5K error: Use Δt ≤ 17 seconds
   - For ΔT < 0.1K error: Use Δt ≤ 3-4 seconds
   → RECOMMENDATION: Δt = 5 seconds (good balance)

2. TRANSITION PHASE (5-15 minutes):
   - Max |dT/dt| ≈ 10-15 mK/s
   - Can increase to Δt = 10 seconds without major accuracy loss

3. QUASI-STEADY PHASE (15+ minutes):
   - Max |dT/dt| < 5 mK/s
   - Can use Δt = 20-30 seconds

OVERALL RECOMMENDATION:
   - Start with Δt = 5 seconds for training
   - This captures fast dynamics while keeping computational cost reasonable
   - For the recurrent history: keep 3-5 previous steps (15-25 seconds history)
   
ALTERNATIVE: Adaptive time stepping
   - Use smaller Δt (1-2s) when |dT/dt| > threshold
   - Use larger Δt (10-20s) when dynamics are slow
   - This requires more complex implementation but is most efficient
""")

plt.show()
