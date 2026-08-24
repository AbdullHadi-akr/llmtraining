from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    nb_path = Path("battery_surrogate_agenticWorkflow/notebooks/mlp_pointwise_demo.ipynb")
    nb = nbformat.read(nb_path.open("r", encoding="utf-8"), as_version=4)

    code_idxs = [i for i, c in enumerate(nb.cells) if c.get("cell_type") == "code"]
    print(f"TOTAL_CODE_CELLS={len(code_idxs)}", flush=True)

    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
        allow_errors=False,
    )

    with client.setup_kernel():
        for k, i in enumerate(code_idxs, start=1):
            print(f"RUNNING_CELL={k}/{len(code_idxs)}", flush=True)
            client.execute_cell(nb.cells[i], i)

    nbformat.write(nb, nb_path.open("w", encoding="utf-8"))
    print(f"EXECUTED_NOTEBOOK={nb_path}", flush=True)
    print("RUN_COMPLETE=1", flush=True)


if __name__ == "__main__":
    main()
