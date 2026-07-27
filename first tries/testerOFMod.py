"""Minimal Modulus PINN on CPU.

Solves u''(x) + pi^2*sin(pi*x) = 0 on x in [0, 1]
with boundary conditions u(0)=0 and u(1)=0.
Exact solution is u(x)=sin(pi*x).
"""

import math

import torch
from modulus.models.mlp import FullyConnected


def pde_residual(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Compute residual r(x) = u_xx + pi^2*sin(pi*x)."""
    u = model(x)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]
    forcing = (math.pi**2) * torch.sin(math.pi * x)
    return u_xx + forcing


def train_pinn(epochs: int = 1200, n_interior: int = 128, lr: float = 1e-3) -> None:
    device = torch.device("cpu")
    torch.manual_seed(42)

    model = FullyConnected(
        in_features=1,
        out_features=1,
        layer_size=64,
        num_layers=4,
        activation_fn="silu",
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    x_bc = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=device)
    y_bc = torch.zeros_like(x_bc)

    print("Start training Modulus PINN on CPU...")
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Random interior points each epoch for PDE collocation.
        x_interior = torch.rand(n_interior, 1, dtype=torch.float32, device=device)
        x_interior.requires_grad_(True)

        res = pde_residual(model, x_interior)
        loss_pde = torch.mean(res**2)

        pred_bc = model(x_bc)
        loss_bc = torch.mean((pred_bc - y_bc) ** 2)

        loss = loss_pde + 10.0 * loss_bc
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"epoch={epoch:4d} loss={loss.item():.6e} "
                f"pde={loss_pde.item():.6e} bc={loss_bc.item():.6e}"
            )

    model.eval()
    x_test = torch.linspace(0.0, 1.0, 5, dtype=torch.float32, device=device).view(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test)
        u_true = torch.sin(math.pi * x_test)

    print("\nSample predictions:")
    for x_val, pred, true in zip(x_test, u_pred, u_true):
        print(
            f"x={x_val.item():.2f} pred={pred.item(): .6f} "
            f"true={true.item(): .6f} err={abs(pred.item()-true.item()):.3e}"
        )


if __name__ == "__main__":
    train_pinn()