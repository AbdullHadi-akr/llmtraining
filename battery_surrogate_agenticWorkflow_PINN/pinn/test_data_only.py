#!/usr/bin/env python3
"""Simple data-only training test (no physics).

This tests the basic setup: data loading, model, optimizer.
"""

import sys
from pathlib import Path


def main():
    sys.stdout.reconfigure(line_buffering=True)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    import torch
    import numpy as np
    
    print("=" * 60, flush=True)
    print("DATA-ONLY TRAINING TEST", flush=True)
    print("=" * 60, flush=True)
    
    # Load data
    print("\nLoading data...", flush=True)
    project_root = Path(__file__).parent.parent.parent  # battery_surrogate_agenticWorkflow_PINN -> batterysurrogatemodell
    
    from pinn.data.load_op01 import load_op01_data
    from pinn.models.net_T import NetT
    
    op01_data = load_op01_data(
        npz_path=str(project_root / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz"),
        heat_source_csv=str(project_root / "battery_surrogate_agenticWorkflow_PINN/data/OP01_raw/OP01/OP1_Heat Source.csv"),
        subsample_time=10,
    )
    
    t = op01_data["t"]
    xyz = op01_data["xyz"]
    T = op01_data["T"]
    config = op01_data["config"]
    
    n_t = len(t)
    n_points = xyz.shape[0]
    print(f"  Timesteps: {n_t}, Grid points: {n_points}", flush=True)
    print(f"  t: [{t.min():.1f}, {t.max():.1f}] s", flush=True)
    print(f"  T: [{T.min():.2f}, {T.max():.2f}] °C", flush=True)
    
    # Convert to tensors
    device = torch.device("cpu")
    xyz_t = torch.tensor(xyz, dtype=torch.float32, device=device)
    t_arr = torch.tensor(t, dtype=torch.float32, device=device)
    T_t = torch.tensor(T, dtype=torch.float32, device=device)
    
    # Normalization
    xyz_min = xyz_t.min(dim=0).values
    xyz_max = xyz_t.max(dim=0).values
    xyz_norm = (xyz_t - xyz_min) / (xyz_max - xyz_min + 1e-6)  # [0, 1]
    t_norm = t_arr / t_arr.max()  # [0, 1]
    T_init = 25.0
    T_scale = 10.0  # Expected temperature rise
    
    # Config tensor (normalized)
    config_values = torch.tensor([
        config["c_rate"] / 3,
        config["cell_current"] / 1000,
        (config["fluid_initial_temp"] - 25) / 10,
        (config["fluid_inlet_temp"] - 25) / 10,
        config["fluid_mass_flow"] * 1000,
        config["soc_start"] / 100,
        (config["solid_initial_temp"] - 25) / 10,
    ], dtype=torch.float32, device=device)
    
    print(f"\nNormalization:", flush=True)
    print(f"  xyz normalized: [{xyz_norm.min():.2f}, {xyz_norm.max():.2f}]", flush=True)
    print(f"  t normalized: [{t_norm.min():.2f}, {t_norm.max():.2f}]", flush=True)
    print(f"  Config (normalized): {config_values.numpy()}", flush=True)
    
    # Create model
    k = 2
    model = NetT(depth=2, width=64, k=k, n_config=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"\nModel: depth=2, width=64, params={sum(p.numel() for p in model.parameters())}", flush=True)
    
    # Training loop
    print("\nTraining (data-only)...", flush=True)
    batch_size = 16
    epochs = 20
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Sample random timesteps
        t_idx = np.random.choice(n_t, batch_size, replace=False)
        
        total_loss = torch.tensor(0.0, device=device)
        for ti in t_idx:
            x_in = xyz_norm[:, 0:1]
            y_in = xyz_norm[:, 1:2]
            z_in = xyz_norm[:, 2:3]
            t_in = t_norm[ti].expand(n_points, 1)
            cfg = config_values.unsqueeze(0).expand(n_points, -1)
            
            # History (use labels from previous timesteps, normalized)
            if ti >= k:
                hist = (T_t[ti-k:ti].T - T_init) / T_scale  # (n_points, k)
            else:
                hist = torch.zeros(n_points, k, device=device)
            
            inputs = torch.cat([x_in, y_in, z_in, t_in, cfg, hist], dim=-1)
            
            # Forward
            T_pred = model(inputs).squeeze(-1)
            
            # Target (normalized)
            T_target = (T_t[ti] - T_init) / T_scale
            
            # Loss
            loss = torch.mean((T_pred - T_target) ** 2)
            total_loss = total_loss + loss
        
        total_loss = total_loss / batch_size
        total_loss.backward()
        optimizer.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4e}", flush=True)
    
    # Final evaluation
    print("\nFinal evaluation...", flush=True)
    model.eval()
    with torch.no_grad():
        ti = n_t // 2
        x_in = xyz_norm[:, 0:1]
        y_in = xyz_norm[:, 1:2]
        z_in = xyz_norm[:, 2:3]
        t_in = t_norm[ti].expand(n_points, 1)
        cfg = config_values.unsqueeze(0).expand(n_points, -1)
        hist = (T_t[ti-k:ti].T - T_init) / T_scale if ti >= k else torch.zeros(n_points, k)
        
        inputs = torch.cat([x_in, y_in, z_in, t_in, cfg, hist], dim=-1)
        T_pred_norm = model(inputs).squeeze(-1)
        
        # Denormalize
        T_pred = T_pred_norm * T_scale + T_init
        T_label = T_t[ti]
        
        mae = torch.mean(torch.abs(T_pred - T_label)).item()
        mse = torch.mean((T_pred - T_label) ** 2).item()
        
        print(f"  t={t_arr[ti].item():.1f}s", flush=True)
        print(f"  Pred: [{T_pred.min():.2f}, {T_pred.max():.2f}] °C", flush=True)
        print(f"  Label: [{T_label.min():.2f}, {T_label.max():.2f}] °C", flush=True)
        print(f"  MAE: {mae:.3f} °C, MSE: {mse:.3f}", flush=True)
    
    print("\n" + "=" * 60, flush=True)
    print("DATA-ONLY TEST COMPLETE", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
