"""Progress printing utilities for training loops."""

from __future__ import annotations

from typing import Callable


def make_progress_printer(total: int, desc: str = "", use_tqdm: bool = True) -> Callable:
    """
    Create a progress callback for training loops.

    Parameters
    ----------
    total : int
        Total number of iterations (e.g., epochs)
    desc : str
        Description prefix (e.g., "Training")
    use_tqdm : bool
        Try to use tqdm if available, else fall back to print

    Returns
    -------
    Callable
        Progress callback: cb(done, total=None, msg="")
    """
    try:
        if use_tqdm:
            from tqdm import tqdm
            pbar = tqdm(total=total, desc=desc, unit="it", dynamic_ncols=True)
            
            def cb(done: int, total_arg: int | None = None, msg: str = "") -> None:
                # Update bar to current position
                current_pos = pbar.n
                if done > current_pos:
                    pbar.update(done - current_pos)
                if msg:
                    pbar.set_postfix_str(msg)
            
            return cb
    except ImportError:
        pass
    
    # Fallback to simple print
    def cb_print(done: int, total_arg: int | None = None, msg: str = "") -> None:
        pct = 100.0 * done / total if total else 0
        prefix = f"{desc} " if desc else ""
        suffix = f" | {msg}" if msg else ""
        print(f"\r{prefix}{done}/{total} ({pct:.0f}%){suffix}", end="", flush=True)
        if done == total:
            print()  # newline at end
    
    return cb_print
