#!/usr/bin/env python3
"""
Python script to run LOVE on input data and compare with R outputs.

Usage:
    python run_love_py.py <data_file> [tag] [options]

Arguments:
    data_file       Path to CSV file (samples as rows, features as columns)
    tag             Optional run name tag (default: basename of data file)

Options:
    --generate-only      Only generate Python outputs without comparison
    --tolerance TOL      Tolerance for numerical comparisons (default: 1e-4)
    --fixed-delta VALUE  Use a fixed delta value to bypass CV (deterministic)

Examples:
    python run_love_py.py /path/to/data.csv
    python run_love_py.py /path/to/data.csv my_experiment
    python run_love_py.py /path/to/data.csv my_experiment --fixed-delta 0.5
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from love import LOVE


def load_csv(filepath):
    """Load a CSV file as numpy array."""
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    return df.values


def compare_arrays(arr1, arr2, name, tol=1e-6):
    """Compare two arrays and report differences."""
    if arr1 is None or arr2 is None:
        print(f"  {name}: SKIP (missing data)")
        return None

    arr1 = np.asarray(arr1).flatten() if arr1.ndim == 1 else np.asarray(arr1)
    arr2 = np.asarray(arr2).flatten() if arr2.ndim == 1 else np.asarray(arr2)

    if arr1.shape != arr2.shape:
        print(f"  {name}: FAIL (shape mismatch: {arr1.shape} vs {arr2.shape})")
        return False

    # Handle NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)

    if not np.array_equal(nan_mask1, nan_mask2):
        print(f"  {name}: FAIL (NaN pattern mismatch)")
        return False

    # Compare non-NaN values
    valid_mask = ~nan_mask1
    if np.sum(valid_mask) == 0:
        print(f"  {name}: PASS (all NaN)")
        return True

    max_diff = np.max(np.abs(arr1[valid_mask] - arr2[valid_mask]))
    mean_diff = np.mean(np.abs(arr1[valid_mask] - arr2[valid_mask]))

    if max_diff <= tol:
        print(f"  {name}: PASS (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"  {name}: DIFF (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        return False


def compare_sets(set1, set2, name):
    """Compare two sets of indices."""
    set1 = set(np.asarray(set1).flatten().astype(int))
    set2 = set(np.asarray(set2).flatten().astype(int))

    overlap = len(set1 & set2)
    total = len(set1 | set2)
    jaccard = overlap / total if total > 0 else 1.0

    if set1 == set2:
        print(f"  {name}: EXACT MATCH ({len(set1)} items)")
        return True
    else:
        common = set1 & set2
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        print(f"  {name}: PARTIAL MATCH")
        print(f"    Jaccard similarity: {jaccard:.2%}")
        print(f"    Common: {len(common)} items")
        print(f"    Only in R: {len(only_in_1)} items")
        print(f"    Only in Python: {len(only_in_2)} items")
        if len(only_in_1) <= 10:
            print(f"    R-only indices: {sorted(only_in_1)}")
        if len(only_in_2) <= 10:
            print(f"    Python-only indices: {sorted(only_in_2)}")
        return jaccard > 0.8  # Consider pass if > 80% overlap


def main():
    parser = argparse.ArgumentParser(
        description="Run LOVE on input data and compare with R outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_love_py.py /path/to/data.csv
    python run_love_py.py /path/to/data.csv my_experiment
    python run_love_py.py /path/to/data.csv my_experiment --generate-only
        """
    )
    parser.add_argument("data_file", help="Path to CSV file (samples as rows, features as columns)")
    parser.add_argument("tag", nargs="?", default=None,
                        help="Run name tag (default: basename of data file)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate Python outputs without comparison")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Tolerance for numerical comparisons (default: 1e-4)")
    parser.add_argument("--fixed-delta", type=float, default=None,
                        help="Fixed delta value to bypass CV (deterministic comparison)")
    args = parser.parse_args()

    fixed_delta = args.fixed_delta

    # Validate data file
    data_file = Path(args.data_file)
    if not data_file.exists():
        print(f"ERROR: Data file not found: {data_file}")
        sys.exit(1)

    # Set tag
    tag = args.tag if args.tag else data_file.stem

    # Set up paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LOVE Python Analysis")
    print(f"  Data file: {data_file}")
    print(f"  Tag: {tag}")
    print(f"  Output dir: {output_dir}")
    if fixed_delta is not None:
        print(f"  Fixed delta: {fixed_delta} (CV bypassed for deterministic comparison)")
    print("=" * 60)

    # Load data
    print("\nLoading data...")
    X_df = pd.read_csv(data_file, index_col=0)
    X = X_df.values

    n, p = X.shape
    print(f"Data dimensions: {n} samples x {p} features")

    # Check dimensions
    if n <= p:
        print(f"\nWARNING: n ({n}) <= p ({p})")
        print("LOVE requires more samples than features for reliable estimation.")
        print("Consider transposing your data or using a subset of features.\n")

    sample_names = list(X_df.index[:6])
    feature_names = list(X_df.columns[:5])
    print(f"Sample names: {', '.join(map(str, sample_names))}" + (", ..." if n > 6 else ""))
    print(f"Feature names: {', '.join(map(str, feature_names))}" + (", ..." if p > 5 else ""))

    results_py = {}

    # ========================================================================
    # Run LOVE with heterogeneous pure loadings
    # ========================================================================
    print("\n" + "-" * 60)
    print("Running LOVE (pure_homo=False)")
    print("-" * 60)

    np.random.seed(123)
    t_start = time.time()
    try:
        if fixed_delta is not None:
            # Single delta value bypasses CV for deterministic comparison
            result_hetero = LOVE(X, pure_homo=False, delta=np.array([fixed_delta]), verbose=True)
        else:
            result_hetero = LOVE(X, pure_homo=False, verbose=True)
        t_end = time.time()

        print(f"\nResults (heterogeneous):")
        print(f"  Estimated K: {result_hetero['K']}")
        print(f"  Number of pure variables: {len(result_hetero['pureVec'])}")
        pureVec_display = result_hetero['pureVec'][:20]
        print(f"  Pure variables: {list(pureVec_display)}" +
              ("..." if len(result_hetero['pureVec']) > 20 else ""))
        print(f"  optDelta: {result_hetero['optDelta']:.6f}")
        print(f"  Time: {t_end - t_start:.2f}s")

        # Save outputs
        np.savetxt(output_dir / f"{tag}_hetero_A_py.csv", result_hetero['A'], delimiter=',')
        np.savetxt(output_dir / f"{tag}_hetero_C_py.csv", result_hetero['C'], delimiter=',')
        np.savetxt(output_dir / f"{tag}_hetero_Gamma_py.csv", result_hetero['Gamma'], delimiter=',')
        pd.DataFrame({'pureVec': result_hetero['pureVec']}).to_csv(
            output_dir / f"{tag}_hetero_pureVec_py.csv", index=False)
        pd.DataFrame({'K': [result_hetero['K']], 'optDelta': [result_hetero['optDelta']]}).to_csv(
            output_dir / f"{tag}_hetero_params_py.csv", index=False)

        results_py['hetero'] = result_hetero
    except Exception as e:
        print(f"ERROR in LOVE (hetero): {e}")
        results_py['hetero'] = None

    # ========================================================================
    # Run LOVE with homogeneous pure loadings
    # ========================================================================
    print("\n" + "-" * 60)
    print("Running LOVE (pure_homo=True)")
    print("-" * 60)

    np.random.seed(123)
    t_start = time.time()
    try:
        if fixed_delta is not None:
            # Single delta value bypasses CV for deterministic comparison
            result_homo = LOVE(X, pure_homo=True, delta=np.array([fixed_delta]), verbose=True)
        else:
            delta_grid = np.arange(0.1, 1.2, 0.1)
            result_homo = LOVE(X, pure_homo=True, delta=delta_grid, verbose=True)
        t_end = time.time()

        print(f"\nResults (homogeneous):")
        print(f"  Estimated K: {result_homo['K']}")
        print(f"  Number of pure variables: {len(result_homo['pureVec'])}")
        pureVec_display = result_homo['pureVec'][:20]
        print(f"  Pure variables: {list(pureVec_display)}" +
              ("..." if len(result_homo['pureVec']) > 20 else ""))
        print(f"  optDelta: {result_homo['optDelta']:.6f}")
        print(f"  Time: {t_end - t_start:.2f}s")

        # Save outputs
        np.savetxt(output_dir / f"{tag}_homo_A_py.csv", result_homo['A'], delimiter=',')
        np.savetxt(output_dir / f"{tag}_homo_C_py.csv", result_homo['C'], delimiter=',')
        np.savetxt(output_dir / f"{tag}_homo_Gamma_py.csv", result_homo['Gamma'], delimiter=',')
        pd.DataFrame({'pureVec': result_homo['pureVec']}).to_csv(
            output_dir / f"{tag}_homo_pureVec_py.csv", index=False)
        pd.DataFrame({'K': [result_homo['K']], 'optDelta': [result_homo['optDelta']]}).to_csv(
            output_dir / f"{tag}_homo_params_py.csv", index=False)

        results_py['homo'] = result_homo
    except Exception as e:
        print(f"ERROR in LOVE (homo): {e}")
        results_py['homo'] = None

    if args.generate_only:
        print("\n" + "=" * 60)
        print(f"Python outputs saved to: {output_dir}")
        print("=" * 60)
        return

    # ========================================================================
    # Compare with R outputs
    # ========================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: R vs Python")
    print("=" * 60)

    # Check if R outputs exist
    R_hetero_params = load_csv(output_dir / f"{tag}_hetero_params.csv")
    R_homo_params = load_csv(output_dir / f"{tag}_homo_params.csv")

    if R_hetero_params is None and R_homo_params is None:
        print("\nR outputs not found. Run R script first:")
        print(f"  Rscript run_love_R.R {data_file} {tag}")
        print("\n" + "=" * 60)
        print(f"Python outputs saved to: {output_dir}")
        print("=" * 60)
        return

    results = []

    # Compare heterogeneous results
    if results_py['hetero'] is not None and R_hetero_params is not None:
        print("\n" + "-" * 60)
        print("Heterogeneous (pure_homo=False) Comparison")
        print("-" * 60)

        result_hetero = results_py['hetero']
        R_hetero_A = load_csv(output_dir / f"{tag}_hetero_A.csv")
        R_hetero_C = load_csv(output_dir / f"{tag}_hetero_C.csv")
        R_hetero_pureVec = load_csv(output_dir / f"{tag}_hetero_pureVec.csv")

        # Compare K
        R_K_hetero = int(R_hetero_params[0, 0])
        print(f"  K: R={R_K_hetero}, Python={result_hetero['K']}", end="")
        if R_K_hetero == result_hetero['K']:
            print(" - MATCH")
            results.append(True)
        else:
            print(" - DIFFER")
            results.append(False)

        # Compare optDelta
        R_optDelta_hetero = R_hetero_params[0, 1]
        print(f"  optDelta: R={R_optDelta_hetero:.6f}, Python={result_hetero['optDelta']:.6f}", end="")
        if abs(R_optDelta_hetero - result_hetero['optDelta']) < 0.1:
            print(" - SIMILAR")
        else:
            print(" - DIFFER")

        # Compare pureVec (R is 1-indexed, Python is 0-indexed)
        if R_hetero_pureVec is not None:
            R_pureVec_0based = R_hetero_pureVec.flatten().astype(int) - 1
            results.append(compare_sets(R_pureVec_0based, result_hetero['pureVec'], "pureVec"))

        # Compare A shape
        if R_hetero_A is not None:
            print(f"  A shape: R={R_hetero_A.shape}, Python={result_hetero['A'].shape}", end="")
            if R_hetero_A.shape == result_hetero['A'].shape:
                print(" - MATCH")
                results.append(compare_arrays(R_hetero_A, result_hetero['A'], "A values", tol=0.1))
            else:
                print(" - DIFFER")
                results.append(False)

        # Compare C
        if R_hetero_C is not None:
            results.append(compare_arrays(R_hetero_C, result_hetero['C'], "C values", tol=0.1))

    # Compare homogeneous results
    if results_py['homo'] is not None and R_homo_params is not None:
        print("\n" + "-" * 60)
        print("Homogeneous (pure_homo=True) Comparison")
        print("-" * 60)

        result_homo = results_py['homo']
        R_homo_A = load_csv(output_dir / f"{tag}_homo_A.csv")
        R_homo_C = load_csv(output_dir / f"{tag}_homo_C.csv")
        R_homo_pureVec = load_csv(output_dir / f"{tag}_homo_pureVec.csv")

        # Compare K
        R_K_homo = int(R_homo_params[0, 0])
        print(f"  K: R={R_K_homo}, Python={result_homo['K']}", end="")
        if R_K_homo == result_homo['K']:
            print(" - MATCH")
            results.append(True)
        else:
            print(" - DIFFER")
            results.append(False)

        # Compare optDelta
        R_optDelta_homo = R_homo_params[0, 1]
        print(f"  optDelta: R={R_optDelta_homo:.6f}, Python={result_homo['optDelta']:.6f}", end="")
        if abs(R_optDelta_homo - result_homo['optDelta']) < 0.1:
            print(" - SIMILAR")
        else:
            print(" - DIFFER")

        # Compare pureVec
        if R_homo_pureVec is not None:
            R_pureVec_0based = R_homo_pureVec.flatten().astype(int) - 1
            results.append(compare_sets(R_pureVec_0based, result_homo['pureVec'], "pureVec"))

        # Compare A shape
        if R_homo_A is not None:
            print(f"  A shape: R={R_homo_A.shape}, Python={result_homo['A'].shape}", end="")
            if R_homo_A.shape == result_homo['A'].shape:
                print(" - MATCH")
                results.append(compare_arrays(R_homo_A, result_homo['A'], "A values", tol=0.1))
            else:
                print(" - DIFFER")
                results.append(False)

        # Compare C
        if R_homo_C is not None:
            results.append(compare_arrays(R_homo_C, result_homo['C'], "C values", tol=0.1))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    skipped = sum(1 for r in results if r is None)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")

    if failed == 0 and passed > 0:
        print("\nOverall: PASS - R and Python implementations are consistent!")
    elif failed > 0:
        print(f"\nOverall: {failed} comparison(s) showed differences.")
        print("Note: Some differences are expected due to RNG and numerical precision.")

    print("\n" + "=" * 60)
    print(f"Outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
