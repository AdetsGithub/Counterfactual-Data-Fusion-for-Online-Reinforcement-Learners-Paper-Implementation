"""Greedy Casino simulation comparing Standard MAB, RDC, and Data-Fusion RDC agents."""

import argparse

from core.plotting import plot_results


def main():
    parser = argparse.ArgumentParser(description="Run Greedy Casino simulation.")
    parser.add_argument(
        "--mode",
        choices=["original", "generalized"],
        required=True,
        help="Scenario: original 4x4 or generalized KxK",
    )
    parser.add_argument(
        "-k", "--k", type=int, default=20, help="Arms/intents (generalized only)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Total steps T (default: 50000 original, 400*k^2 generalized)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", default="results.png", help="Output plot path"
    )
    args = parser.parse_args()

    if args.mode == "original":
        from scenarios.original.run import run_simulation

        T = args.steps if args.steps is not None else 50_000
        histories = run_simulation(T=T, seed=args.seed)
    else:
        from scenarios.generalized.run import run_simulation

        T = args.steps if args.steps is not None else int(400 * (args.k ** 2))
        histories = run_simulation(k=args.k, T=T, seed=args.seed)

    plot_results(histories, output=args.output)


if __name__ == "__main__":
    main()
