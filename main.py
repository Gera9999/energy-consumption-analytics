from __future__ import annotations

import argparse
from pathlib import Path

from dashboard.plotly_dashboard import save_dashboard_html
from src.pipeline import PipelineConfig, run_pipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the energy analytics pipeline and generate an interactive dashboard.")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the CSV dataset (defaults to data/household_power_consumption.csv).",
    )
    p.add_argument(
        "--demo",
        action="store_true",
        help="Run using the bundled sample dataset (no private dataset required).",
    )
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output HTML path (default: output/dashboard.html).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent

    default_dataset = repo_root / "data" / "household_power_consumption.csv"
    demo_dataset = repo_root / "data" / "sample_household_power_consumption.csv"

    if args.demo:
        csv_path = demo_dataset
    elif args.dataset:
        csv_path = Path(args.dataset)
    else:
        csv_path = default_dataset if default_dataset.exists() else demo_dataset

    if args.output:
        output_html = Path(args.output)
    else:
        output_html = repo_root / "output" / ("dashboard_demo.html" if args.demo else "dashboard.html")

    result = run_pipeline(PipelineConfig(csv_path=csv_path))
    saved_path = save_dashboard_html(
        processed=result.processed,
        forecast=result.forecast,
        output_path=output_html,
    )

    print(f"Dashboard generated: {saved_path}")


if __name__ == "__main__":
    main()
