#!/usr/bin/env python3
"""Run OP01 Battery PINN training.

This script orchestrates the full PINN training pipeline:
1. Load OP01 data
2. Load material properties
3. Load inlet/outlet faces
4. Train temperature PINN (net_T)
5. Train voltage network (net_V)
6. Save results and loss history

Usage:
    cd /mnt/c/Users/M0245635/batterysurrogatemodell
    source modulus_env/bin/activate
    python3 battery_surrogate_agenticWorkflow_PINN/pinn/run_op01.py

Architecture sweep (M3.5):
    python3 battery_surrogate_agenticWorkflow_PINN/pinn/run_op01.py --sweep
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add battery_surrogate_agenticWorkflow_PINN to path so "pinn" can be imported
PINN_ROOT = Path(__file__).parent.parent  # battery_surrogate_agenticWorkflow_PINN
PROJECT_ROOT = PINN_ROOT.parent  # batterysurrogatemodell
sys.path.insert(0, str(PINN_ROOT))

from pinn.data.load_op01 import load_op01_data
from pinn.data.load_properties import load_material_properties
from pinn.data.load_faces import load_inlet_outlet_faces
from pinn.train.solve_T import train_temperature_pinn, LossComponents
from pinn.train.train_V import train_voltage_network


def load_all_data(project_root: Path, subsample_time: int = 10) -> Dict:
    """Load all data needed for training.
    
    Args:
        project_root: Path to project root
        subsample_time: Subsample every N timesteps
        
    Returns:
        Dictionary with all loaded data
    """
    print("=" * 80)
    print("Loading data...")
    print("=" * 80)
    
    # Load OP01 data
    op01_data = load_op01_data(
        npz_path=str(project_root / "battery_surrogate_agenticWorkflow/data_cache/OP01.npz"),
        heat_source_csv=str(project_root / "battery_surrogate_agenticWorkflow_PINN/data/OP01_raw/OP01/OP1_Heat Source.csv"),
        subsample_time=subsample_time,
    )
    print(f"  OP01 data: t shape={op01_data['t'].shape}, T shape={op01_data['T'].shape}")
    print(f"  Config: {op01_data['config']}")
    
    # Load material properties
    props = load_material_properties(
        layer=op01_data["layer"],
        props_cc_dir=str(project_root / "battery_surrogate_agenticWorkflow_PINN/Cell Center"),
        props_jr1_dir=str(project_root / "battery_surrogate_agenticWorkflow_PINN/JR1 Center"),
    )
    print(f"  Material props: rho range=[{props['rho'].min():.1f}, {props['rho'].max():.1f}]")
    
    # Load faces
    faces = load_inlet_outlet_faces(
        str(project_root / "battery_surrogate_agenticWorkflow_PINN/data/inlet_outlet_faces.csv")
    )
    print(f"  Faces: inlet={faces['inlet_xyz'].shape[0]}, outlet={faces['outlet_xyz'].shape[0]}")
    
    return {
        "op01": op01_data,
        "props": props,
        "faces": faces,
    }


def run_temperature_training(
    data: Dict,
    depth: int = 4,
    width: int = 128,
    epochs: int = 500,
    k: int = 2,
    hard_ic: bool = True,
    batch_size: int = 32,
    log_interval: int = 50,
    output_dir: str = None,
) -> Tuple:
    """Run temperature PINN training.
    
    Returns:
        (model, loss_history)
    """
    print("\n" + "=" * 80)
    print(f"Training Temperature PINN (depth={depth}, width={width})")
    print("=" * 80)
    
    op01 = data["op01"]
    props = data["props"]
    faces = data["faces"]
    
    # Get BC temperatures
    T_inlet = op01["config"]["fluid_inlet_temp"]
    T_outlet = op01["config"]["fluid_inlet_temp"]  # Use inlet temp as proxy for outlet
    
    model, history = train_temperature_pinn(
        t=op01["t"],
        xyz=op01["xyz"],
        T_labels=op01["T"],
        layer=op01["layer"],
        config=op01["config"],
        q_dot=op01["q_dot"],
        rho=props["rho"],
        Cp=props["Cp"],
        lambda_tensor=props["lambda_tensor"],
        inlet_xyz=faces["inlet_xyz"],
        outlet_xyz=faces["outlet_xyz"],
        T_inlet=T_inlet,
        T_outlet=T_outlet,
        epochs=epochs,
        depth=depth,
        width=width,
        k=k,
        hard_ic=hard_ic,
        batch_size=batch_size,
        log_interval=log_interval,
        output_dir=output_dir,
    )
    
    return model, history


def run_voltage_training(
    data: Dict,
    depth: int = 2,
    width: int = 64,
    epochs: int = 500,
    log_interval: int = 50,
    output_dir: str = None,
) -> Tuple:
    """Run voltage network training.
    
    Returns:
        (model, loss_history)
    """
    print("\n" + "=" * 80)
    print(f"Training Voltage Network (depth={depth}, width={width})")
    print("=" * 80)
    
    op01 = data["op01"]
    
    model, history = train_voltage_network(
        t=op01["t"],
        bc_V=op01["bc_V"],
        config=op01["config"],
        epochs=epochs,
        depth=depth,
        width=width,
        log_interval=log_interval,
        output_dir=output_dir,
    )
    
    return model, history


def run_architecture_sweep(data: Dict, epochs: int = 500, with_physics: bool = False) -> None:
    """Run architecture sweep (M3.5).
    
    Tests three configurations:
    - small: depth=2, width=64
    - medium: depth=4, width=128
    - large: depth=6, width=256
    
    Args:
        data: Loaded data dict
        epochs: Number of training epochs
        with_physics: If True, include physics loss
    """
    from pinn.train.solve_T_simple import train_simplified, LossComponents
    
    mode = "Physics+Data" if with_physics else "Data-Only"
    print("\n" + "=" * 80)
    print(f"ARCHITECTURE SWEEP (M3.5) - {mode} Training")
    print("=" * 80)
    
    configs = [
        {"name": "small", "depth": 2, "width": 64},
        {"name": "medium", "depth": 4, "width": 128},
        {"name": "large", "depth": 6, "width": 256},
    ]
    
    op01 = data["op01"]
    props = data["props"]
    results = []
    
    # Custom print function that flushes
    def print_flush(msg):
        print(msg, flush=True)
    
    for cfg in configs:
        print_flush(f"\n--- Running {cfg['name']}: depth={cfg['depth']}, width={cfg['width']} ---")
        print_flush("")
        print_flush("=" * 80)
        print_flush(f"Training Temperature PINN (depth={cfg['depth']}, width={cfg['width']})")
        print_flush("=" * 80)
        
        # Physics params if enabled
        phys_kwargs = {}
        if with_physics:
            # Compute effective lambda (isotropic approx: trace/3)
            lambda_tensor = props["lambda_tensor"]  # (n_points, 3, 3)
            lambda_eff = (lambda_tensor[:, 0, 0] + lambda_tensor[:, 1, 1] + lambda_tensor[:, 2, 2]) / 3
            
            phys_kwargs = {
                "q_dot": op01["q_dot"],
                "rho": props["rho"],
                "Cp": props["Cp"],
                "lambda_eff": lambda_eff,
                "w_phys": 0.1,
            }
        
        model, history, final = train_simplified(
            t=op01["t"],
            xyz=op01["xyz"],
            T_labels=op01["T"],
            config=op01["config"],
            depth=cfg["depth"],
            width=cfg["width"],
            k=2,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            log_interval=max(1, epochs // 10),  # 10 log lines
            print_fn=print_flush,
            **phys_kwargs,
        )
        
        results.append({
            "config": cfg["name"],
            "depth": cfg["depth"],
            "width": cfg["width"],
            "L_data": final.data,
            "L_phys": final.phys,
            "L_IC": final.ic,
            "L_BC": final.bc_in + final.bc_out,
            "Total": final.total,
            "params": sum(p.numel() for p in model.parameters()),
        })
    
    # Print results table
    print_flush("\n" + "=" * 80)
    print_flush("ARCHITECTURE SWEEP RESULTS")
    print_flush("=" * 80)
    print_flush(f"{'Config':<10} {'depth':>6} {'width':>6} {'params':>10} {'L_data':>12} {'L_phys':>12} {'L_IC':>12} {'L_BC':>12} {'Total':>12}")
    print_flush("-" * 80)
    for r in results:
        print_flush(
            f"{r['config']:<10} {r['depth']:>6} {r['width']:>6} {r['params']:>10} "
            f"{r['L_data']:>12.4e} {r['L_phys']:>12.4e} {r['L_IC']:>12.4e} "
            f"{r['L_BC']:>12.4e} {r['Total']:>12.4e}"
        )
    print_flush("=" * 80)
    
    return results


def main():
    # Enable line buffering for real-time output
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    
    parser = argparse.ArgumentParser(description="Run OP01 Battery PINN")
    parser.add_argument("--sweep", action="store_true", help="Run architecture sweep (M3.5)")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--depth", type=int, default=4, help="Network depth for single run")
    parser.add_argument("--width", type=int, default=128, help="Network width for single run")
    parser.add_argument("--k", type=int, default=2, help="History length k")
    parser.add_argument("--soft-ic", action="store_true", help="Use soft IC instead of hard")
    parser.add_argument("--subsample", type=int, default=10, help="Time subsample factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (timesteps per step)")
    parser.add_argument("--skip-V", action="store_true", help="Skip voltage training")
    args = parser.parse_args()
    
    # Find project root
    project_root = Path(__file__).parent.parent.parent
    print(f"Project root: {project_root}")
    
    # Output directory
    output_dir = project_root / "battery_surrogate_agenticWorkflow_PINN/artifacts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_all_data(project_root, subsample_time=args.subsample)
    
    if args.sweep:
        # Architecture sweep
        run_architecture_sweep(data, epochs=args.epochs)
    else:
        # Single training run
        run_temperature_training(
            data=data,
            depth=args.depth,
            width=args.width,
            epochs=args.epochs,
            k=args.k,
            hard_ic=not args.soft_ic,
            batch_size=args.batch_size,
            log_interval=50,
            output_dir=str(output_dir),
        )
        
        if not args.skip_V:
            run_voltage_training(
                data=data,
                epochs=args.epochs,
                log_interval=50,
                output_dir=str(output_dir),
            )
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
