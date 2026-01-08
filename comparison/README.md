# LOVE R vs Python Comparison

This directory contains scripts to compare the R and Python implementations of LOVE.

## Prerequisites

### R Requirements
- R with packages: `igraph`, `MASS`, `linprog`

Install R packages if needed:
```r
install.packages(c("igraph", "MASS", "linprog"))
```

### Python Requirements
- Python 3.8+
- numpy, scipy, networkx, pandas

Install Python packages if needed:
```bash
pip install numpy scipy networkx pandas
```

## Running the Comparison

### Step 1: Generate R Reference Outputs

```bash
cd /ocean/projects/cis240075p/arosen1/1_misc/LOVE/comparison
Rscript run_r_comparison.R
```

This will:
- Run various LOVE functions on synthetic data
- Save outputs to `outputs/` directory

### Step 2: Compare with Python

```bash
python compare_outputs.py
```

This will:
- Run the same tests in Python
- Compare outputs with R reference values
- Report any differences

## Tests Performed

| Test | Description |
|------|-------------|
| Test 1 | LOVE with heterogeneous pure loadings (pure_homo=FALSE) |
| Test 2 | LOVE with homogeneous pure loadings (pure_homo=TRUE) |
| Test 3 | Screen_X pre-screening |
| Test 4 | Score_mat computation |
| Test 5 | EstC covariance estimation |
| Test 6 | estOmega precision matrix estimation |

## Expected Differences

Due to:
1. Random number generator differences between R and Python
2. LP solver implementation differences
3. Numerical precision differences

Some numerical differences are expected. The comparison script uses tolerances:
- Default tolerance: 1e-6 for most comparisons
- Larger tolerance (0.1) for LP-based estimations

## Troubleshooting

**R script fails to find source files:**
- Make sure you're running from the `comparison/` directory
- Or provide the full path: `Rscript /path/to/comparison/run_r_comparison.R`

**Python comparison shows differences:**
- Check if R outputs exist in `outputs/` directory
- Small numerical differences are expected due to RNG differences
- Focus on structural comparisons (K, pureVec indices)
