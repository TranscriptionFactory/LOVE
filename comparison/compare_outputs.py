#!/usr/bin/env python3
"""
Python script to compare LOVE outputs with R reference outputs.

Run the R script first (run_r_comparison.R), then run this script.

Usage:
    python compare_outputs.py [--generate-only] [--tolerance 1e-6]
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from love import LOVE, Screen_X, Score_mat, EstC, estOmega


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
        print(f"  {name}: FAIL (max diff: {max_diff:.2e}, mean diff: {mean_diff:.2e})")
        return False


def compare_sets(set1, set2, name):
    """Compare two sets of indices."""
    set1 = set(np.asarray(set1).flatten().astype(int))
    set2 = set(np.asarray(set2).flatten().astype(int))

    if set1 == set2:
        print(f"  {name}: PASS (exact match)")
        return True
    else:
        common = set1 & set2
        only_in_1 = set1 - set2
        only_in_2 = set2 - set1
        print(f"  {name}: FAIL")
        print(f"    Common: {sorted(common)}")
        print(f"    Only in R: {sorted(only_in_1)}")
        print(f"    Only in Python: {sorted(only_in_2)}")
        return False


class ComparisonTester:
    """Class to run comparison tests between R and Python implementations."""

    def __init__(self, output_dir, tolerance=1e-6):
        self.output_dir = Path(output_dir)
        self.tolerance = tolerance
        self.results = {}

    def run_all_tests(self, generate_only=False):
        """Run all comparison tests."""
        print("=" * 60)
        print("LOVE R vs Python Comparison")
        print("=" * 60)

        if not generate_only:
            if not self.output_dir.exists():
                print(f"\nERROR: Output directory not found: {self.output_dir}")
                print("Please run the R script first: Rscript run_r_comparison.R")
                return False

        self.results['test1'] = self.test1_love_hetero(generate_only)
        self.results['test2'] = self.test2_love_homo(generate_only)
        self.results['test3'] = self.test3_screen_x(generate_only)
        self.results['test4'] = self.test4_score_mat(generate_only)
        self.results['test5'] = self.test5_est_c(generate_only)
        self.results['test6'] = self.test6_est_omega(generate_only)

        self.print_summary()
        return all(v is not False for v in self.results.values() if v is not None)

    def test1_love_hetero(self, generate_only=False):
        """Test 1: LOVE with heterogeneous pure loadings."""
        print("\n" + "-" * 60)
        print("Test 1: LOVE (pure_homo=False, using R-generated data)")
        print("-" * 60)

        # Load R-generated data for consistent comparison
        R_X = load_csv(self.output_dir / "test1_X.csv")

        if R_X is None:
            print("  SKIP (no R reference data - run R script first)")
            return None

        X = R_X

        # Run LOVE
        np.random.seed(123)
        result = LOVE(X, pure_homo=False, verbose=False)

        print(f"  Python K: {result['K']}")
        print(f"  Python pureVec: {sorted(result['pureVec'])}")
        print(f"  Python optDelta: {result['optDelta']:.6f}")

        if generate_only:
            # Save Python outputs
            np.savetxt(self.output_dir / "test1_py_A_hat.csv", result['A'], delimiter=',')
            np.savetxt(self.output_dir / "test1_py_C_hat.csv", result['C'], delimiter=',')
            print("  Outputs saved.")
            return None

        # Load R outputs
        R_A_hat = load_csv(self.output_dir / "test1_A_hat.csv")
        R_C_hat = load_csv(self.output_dir / "test1_C_hat.csv")
        R_pureVec = load_csv(self.output_dir / "test1_pureVec.csv")
        R_params = load_csv(self.output_dir / "test1_params.csv")

        if R_params is not None:
            print(f"  R K: {int(R_params[0, 0])}")
            print(f"  R pureVec: {sorted(R_pureVec.flatten().astype(int))}")
            print(f"  R optDelta: {R_params[0, 1]:.6f}")

        # Compare
        results = []
        print("\nComparisons:")

        # Compare K
        if R_params is not None:
            if result['K'] == int(R_params[0, 0]):
                print(f"  K: PASS")
                results.append(True)
            else:
                print(f"  K: FAIL (Python={result['K']}, R={int(R_params[0, 0])})")
                results.append(False)

        # Compare pureVec (as sets, since order may differ)
        if R_pureVec is not None:
            # R uses 1-based indexing, Python uses 0-based
            R_pureVec_0based = R_pureVec.flatten().astype(int) - 1
            results.append(compare_sets(result['pureVec'], R_pureVec_0based, "pureVec"))

        # Compare A (may have sign/column permutation differences)
        # For now, just compare shapes
        if R_A_hat is not None:
            if result['A'].shape == R_A_hat.shape:
                print(f"  A shape: PASS ({result['A'].shape})")
                results.append(True)
            else:
                print(f"  A shape: FAIL (Python={result['A'].shape}, R={R_A_hat.shape})")
                results.append(False)

        return all(results) if results else None

    def test2_love_homo(self, generate_only=False):
        """Test 2: LOVE with homogeneous pure loadings."""
        print("\n" + "-" * 60)
        print("Test 2: LOVE (pure_homo=True, using R-generated data)")
        print("-" * 60)

        # Load R-generated data for consistent comparison
        # Note: Test 2 uses same data structure as test 1 but regenerated
        R_X = load_csv(self.output_dir / "test1_X.csv")

        if R_X is None:
            print("  SKIP (no R reference data - run R script first)")
            return None

        X = R_X

        # Run LOVE
        np.random.seed(123)
        delta_grid = np.arange(0.1, 1.2, 0.1)
        result = LOVE(X, pure_homo=True, delta=delta_grid, verbose=False)

        print(f"  Python K: {result['K']}")
        print(f"  Python pureVec: {sorted(result['pureVec'])}")
        print(f"  Python optDelta: {result['optDelta']:.6f}")

        if generate_only:
            return None

        # Load R outputs
        R_pureVec = load_csv(self.output_dir / "test2_pureVec.csv")
        R_params = load_csv(self.output_dir / "test2_params.csv")

        if R_params is not None:
            print(f"  R K: {int(R_params[0, 0])}")
            print(f"  R pureVec: {sorted(R_pureVec.flatten().astype(int))}")
            print(f"  R optDelta: {R_params[0, 1]:.6f}")

        results = []
        print("\nComparisons:")

        if R_params is not None:
            if result['K'] == int(R_params[0, 0]):
                print(f"  K: PASS")
                results.append(True)
            else:
                print(f"  K: FAIL")
                results.append(False)

        if R_pureVec is not None:
            R_pureVec_0based = R_pureVec.flatten().astype(int) - 1
            results.append(compare_sets(result['pureVec'], R_pureVec_0based, "pureVec"))

        return all(results) if results else None

    def test3_screen_x(self, generate_only=False):
        """Test 3: Screen_X pre-screening."""
        print("\n" + "-" * 60)
        print("Test 3: Screen_X (using R-generated data)")
        print("-" * 60)

        # Load R-generated data for consistent comparison
        R_X = load_csv(self.output_dir / "test3_X.csv")

        if R_X is None:
            print("  SKIP (no R reference data - run R script first)")
            return None

        X = R_X

        # Run Screen_X
        np.random.seed(123)
        result = Screen_X(X)

        print(f"  Python noise_ind: {sorted(result['noise_ind'])}")
        print(f"  Python thresh_min: {result['thresh_min']:.6f}")

        if generate_only:
            return None

        # Load R outputs
        R_noise_ind = load_csv(self.output_dir / "test3_noise_ind.csv")
        R_params = load_csv(self.output_dir / "test3_params.csv")

        if R_noise_ind is not None:
            # R uses 1-based indexing
            R_noise_0based = R_noise_ind.flatten().astype(int) - 1
            print(f"  R noise_ind: {sorted(R_noise_0based)}")

        if R_params is not None:
            print(f"  R thresh_min: {R_params[0, 0]:.6f}")

        results = []
        print("\nComparisons:")

        if R_noise_ind is not None:
            R_noise_0based = R_noise_ind.flatten().astype(int) - 1
            results.append(compare_sets(result['noise_ind'], R_noise_0based, "noise_ind"))

        return all(results) if results else None

    def test4_score_mat(self, generate_only=False):
        """Test 4: Score_mat computation."""
        print("\n" + "-" * 60)
        print("Test 4: Score_mat (using R-generated data)")
        print("-" * 60)

        # Load R-generated correlation matrix for consistent comparison
        R_R = load_csv(self.output_dir / "test4_R.csv")

        if R_R is None:
            print("  SKIP (no R reference data - run R script first)")
            return None

        # Compute score matrix using R's correlation matrix
        result = Score_mat(R_R, q=2, exact=False)

        print(f"  Python score shape: {result['score'].shape}")
        print(f"  Python moments shape: {result['moments'].shape}")

        if generate_only:
            return None

        # Load R outputs
        R_score = load_csv(self.output_dir / "test4_score.csv")
        R_moments = load_csv(self.output_dir / "test4_moments.csv")

        results = []
        print("\nComparisons:")

        if R_score is not None:
            results.append(compare_arrays(result['score'], R_score, "score", self.tolerance))

        if R_moments is not None:
            results.append(compare_arrays(result['moments'], R_moments, "moments", self.tolerance))

        return all(r for r in results if r is not None) if results else None

    def test5_est_c(self, generate_only=False):
        """Test 5: EstC covariance estimation."""
        print("\n" + "-" * 60)
        print("Test 5: EstC (using R-generated data)")
        print("-" * 60)

        # Load R-generated data for consistent comparison
        R_Sigma = load_csv(self.output_dir / "test5_Sigma.csv")
        R_AI = load_csv(self.output_dir / "test5_AI.csv")

        if R_Sigma is None or R_AI is None:
            print("  SKIP (no R reference data - run R script first)")
            return None

        # Estimate C using R's Sigma and AI
        C_est = EstC(R_Sigma, R_AI, diagonal=False)
        C_est_diag = EstC(R_Sigma, R_AI, diagonal=True)

        print(f"  Python C (non-diagonal):\n{C_est}")
        print(f"  Python C (diagonal):\n{C_est_diag}")

        if generate_only:
            return None

        # Load R outputs
        R_C = load_csv(self.output_dir / "test5_C.csv")
        R_C_diag = load_csv(self.output_dir / "test5_C_diag.csv")

        results = []
        print("\nComparisons:")

        if R_C is not None:
            results.append(compare_arrays(C_est, R_C, "C (non-diagonal)", self.tolerance))

        if R_C_diag is not None:
            results.append(compare_arrays(C_est_diag, R_C_diag, "C (diagonal)", self.tolerance))

        return all(r for r in results if r is not None) if results else None

    def test6_est_omega(self, generate_only=False):
        """Test 6: estOmega precision matrix estimation."""
        print("\n" + "-" * 60)
        print("Test 6: estOmega")
        print("-" * 60)

        # Load R's C matrix for consistency
        R_C = load_csv(self.output_dir / "test6_C.csv")
        R_params = load_csv(self.output_dir / "test6_params.csv")

        if R_C is None or R_params is None:
            print("  SKIP (no R reference data)")
            return None

        lbd = R_params[0, 0]

        # Estimate Omega
        Omega_est = estOmega(lbd, R_C)

        print(f"  Python Omega shape: {Omega_est.shape}")
        print(f"  Python Omega:\n{Omega_est}")

        if generate_only:
            return None

        # Load R output
        R_Omega = load_csv(self.output_dir / "test6_Omega.csv")

        results = []
        print("\nComparisons:")

        if R_Omega is not None:
            # Use larger tolerance for LP-based computation
            results.append(compare_arrays(Omega_est, R_Omega, "Omega", tol=0.1))

        return all(r for r in results if r is not None) if results else None

    def print_summary(self):
        """Print summary of all test results."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total = 0
        passed = 0
        failed = 0
        skipped = 0

        for name, result in self.results.items():
            total += 1
            if result is True:
                status = "PASS"
                passed += 1
            elif result is False:
                status = "FAIL"
                failed += 1
            else:
                status = "SKIP"
                skipped += 1
            print(f"  {name}: {status}")

        print("-" * 60)
        print(f"Total: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped}")

        if failed == 0 and passed > 0:
            print("\nAll tests PASSED!")
        elif failed > 0:
            print(f"\n{failed} test(s) FAILED - please investigate.")


def main():
    parser = argparse.ArgumentParser(description="Compare LOVE R and Python outputs")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate Python outputs without comparison")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="Tolerance for numerical comparisons")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"

    if args.generate_only:
        output_dir.mkdir(exist_ok=True)

    tester = ComparisonTester(output_dir, tolerance=args.tolerance)
    success = tester.run_all_tests(generate_only=args.generate_only)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
