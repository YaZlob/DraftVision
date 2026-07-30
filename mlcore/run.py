import os
import argparse
from pathlib import Path
from .solver import SOLVER, BaseSolver
from .core import load_experiment_config, CONFIGNAME


def _dir_path(string) -> Path:
    if not os.path.isdir(string):
        raise ValueError
    return Path(string)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="path/2/config.yml",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=_dir_path,
        help="path/2/output/dir",
    )
    parser.add_argument(
        "-t",
        "--tuning",
        type=str,
        help="absoulte path to model chekpoint.pth",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=_dir_path,
        help="/path/2/prev/exp/dir",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        type=_dir_path,
        help="/path/2/prev/exp/dir",
    )

    args = parser.parse_args()

    if args.config:
        config = load_experiment_config(args.config)
        solver: BaseSolver = SOLVER.get(config.task)(config, args.outdir, args.tuning)
        solver.fit()

    elif args.resume:
        config = load_experiment_config(args.resume / CONFIGNAME)
        solver: BaseSolver = SOLVER.get(config.task)
        solver.resume(args.resume)

    elif args.evaluate:
        config = load_experiment_config(args.evaluate / CONFIGNAME)
        solver: BaseSolver = SOLVER.get(config.task)
        solver.evalualte(args.evaluate)

    else:
        msg = (
            "Provide any: \n"
            "-c config to start trainig from zero\n"
            "-r absolute path to previous experiment to resume it\n"
            "-e absolute path to previous experiment for evaluation\n"
        )
        raise RuntimeError(msg)
