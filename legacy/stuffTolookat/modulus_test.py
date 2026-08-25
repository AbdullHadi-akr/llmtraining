import hydra
from omegaconf import DictConfig

from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_1d import Line1D
from modulus.sym.eq.pdes.diffusion import Diffusion
from modulus.sym.domain.constraint import PointwiseInteriorConstraint
from modulus.sym.models.fully_connected import FullyConnectedArch
from modulus.sym.key import Key


@hydra.main(
    version_base=None,
    config_path="/mnt/c/Users/M0245635/modulus-sym/examples/ldc/conf",
    config_name="config",
)
def run(cfg: DictConfig):

    print("✅ FINAL Modulus test running")

    # PDE
    eq = Diffusion(T="u", D=1.0, dim=1, time=False)

    # Network (fixed!)
    net = FullyConnectedArch(
        input_keys=[Key("x")],
        output_keys=[Key("u")],
        layer_size=16,
        nr_layers=2,
        activation_fn="relu",   # IMPORTANT (avoids crash)
    )

    nodes = [net.make_node(name="net")] + eq.make_nodes()

    # Geometry
    geo = Line1D(0, 1)

    # Domain
    domain = Domain()

    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"diffusion_u": 0},
        batch_size=32,
    )

    domain.add_constraint(interior, "interior")

    # Solver
    solver = Solver(cfg, domain)

    solver.solve()


if __name__ == "__main__":
    run()
