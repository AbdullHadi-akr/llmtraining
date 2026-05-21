from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
#source ~/modulus_cpu/bin/activate für venv dann bash python navier.py
ns = NavierStokes(
    nu=0.01,
    rho=1.0,
    dim=2,
    time=False
)

print("\nNavier-Stokes Gleichungen:\n")

for name, eq in ns.equations.items():
    print(f"{name}:")
    print(eq)
    print()