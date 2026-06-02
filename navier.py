import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# =========================
# Netzwerk
# =========================
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Training Setup
# =========================
model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Trainingspunkte
x = torch.linspace(0, 1, 100).view(-1, 1)
x.requires_grad = True

loss_history = []

# =========================
# Training (PDE: u'' = 0)
# =========================
for step in range(2000):
    optimizer.zero_grad()

    u = model(x)

    # Erste Ableitung
    du_dx = torch.autograd.grad(
        u, x, torch.ones_like(u), create_graph=True
    )[0]

    # Zweite Ableitung
    d2u_dx2 = torch.autograd.grad(
        du_dx, x, torch.ones_like(du_dx), create_graph=True
    )[0]

    # PDE: u'' = 0
    loss_pde = torch.mean(d2u_dx2**2)

    loss = loss_pde
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if step % 100 == 0:
        print(f"step {step}, loss {loss.item():.6f}")



# =========================
# Plot
# =========================
plt.figure()
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("loss.png")
print("Plot gespeichert als loss.png ✅")