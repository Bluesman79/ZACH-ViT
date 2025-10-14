#!/usr/bin/env python3
"""
Command-line interface for ZACH-ViT preprocessing.
Runs all 4 modules: ROI → VIS → 0-SSDA → SSDA_p
"""
import argparse
from zachvit.preprocessing_utils import run_full_preprocessing_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run ZACH-ViT preprocessing pipeline.")
    parser.add_argument("--talos_path", type=str, default="../Data/TALOS",
                        help="Path to TALOS DICOM dataset root.")
    parser.add_argument("--output_dir", type=str, default="../Data",
                        help="Base output directory for processed data.")
    parser.add_argument("--patient_start", type=int, default=100)
    parser.add_argument("--patient_end", type=int, default=122)
    parser.add_argument("--primes", type=int, nargs="+", default=[2, 3],
                        help="List of prime numbers for SSDA_p seeds.")
    parser.add_argument("--num_positions", type=int, default=4)
    args = parser.parse_args()

    run_full_preprocessing_pipeline(
        talos_path=args.talos_path,
        output_dir_roi=f"{args.output_dir}/Processed_ROI",
        output_dir_vis=f"{args.output_dir}/VIS",
        output_dir_0_ssda=f"{args.output_dir}/0_SSDA",
        prime_seeds=args.primes,
        patient_range=(args.patient_start, args.patient_end),
        num_positions=args.num_positions
    )

if __name__ == "__main__":
    main()
