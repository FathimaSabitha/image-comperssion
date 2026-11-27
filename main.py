"""
Main Project Runner
===================
Entrypoint for the Mathematical Image Compression Project.
Allows executing experiments, running mathematical verification tests,
or launching the Streamlit interactive dashboard.
"""

import sys
import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Mathematical Image Compression Using SVD and DCT (MCA Project)"
    )
    parser.add_argument(
        "--mode",
        choices=["app", "test", "experiment", "generate-samples"],
        default="app",
        help="Execution mode: 'app' (Streamlit UI), 'test' (unit tests), 'experiment' (run benchmark suite), 'generate-samples' (create sample images)"
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    python_bin = sys.executable

    if args.mode == "generate-samples":
        print("[*] Generating academic test images...")
        subprocess.run([python_bin, os.path.join(project_root, "sample_images", "generate_samples.py")], check=True)

    elif args.mode == "test":
        print("[*] Running unit tests for SVD, DCT, and statistical evaluation metrics...")
        subprocess.run([python_bin, "-m", "unittest", "discover", "tests"], check=True)

    elif args.mode == "experiment":
        print("[*] Running batch comparative experiments across all sample images...")
        subprocess.run([python_bin, os.path.join(project_root, "experiments", "run_experiments.py")], check=True)

    elif args.mode == "app":
        print("[*] Launching Streamlit Interactive Web Application...")
        cmd = ["streamlit", "run", os.path.join(project_root, "app.py")]
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
