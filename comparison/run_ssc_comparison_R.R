#!/usr/bin/env Rscript
#
# R script to run LOVE on SSc (Systemic Sclerosis) scRNA-seq data
# Run this first, then run run_ssc_comparison_py.py
#
# Usage: Rscript run_ssc_comparison_R.R
#

# Get the directory where this script is located
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

setwd(script_dir)

# Source all R files
r_dir <- file.path(script_dir, "..", "R")
r_files <- c("Utilities.R", "Score.R", "EstPureHomo.R", "EstPureHetero.R",
             "EstNonpure.R", "EstOmega.R", "PreScreen.R", "CV.R", "LOVE.R")

cat("Loading R source files from:", r_dir, "\n")
for (f in r_files) {
  source(file.path(r_dir, f))
}

# Create output directory
output_dir <- file.path(script_dir, "outputs_ssc")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat(rep("=", 60), "\n", sep = "")
cat("LOVE R - SSc Real Data Analysis\n")
cat(rep("=", 60), "\n", sep = "")

# Load SSc data
data_file <- file.path(script_dir, "SSc_X_SLIDE.csv")
cat("\nLoading data from:", data_file, "\n")

X_df <- read.csv(data_file, row.names = 1)
X <- as.matrix(X_df)

cat("Data dimensions: ", nrow(X), " samples x ", ncol(X), " features\n", sep = "")
cat("Sample names: ", paste(head(rownames(X)), collapse = ", "), ", ...\n", sep = "")
cat("Feature names: ", paste(head(colnames(X), 5), collapse = ", "), ", ...\n", sep = "")

# Save the matrix for Python to use (ensures same data)
write.csv(X, file.path(output_dir, "ssc_X_matrix.csv"), row.names = TRUE)

# ============================================================================
# Run LOVE with heterogeneous pure loadings
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Running LOVE (pure_homo = FALSE)\n")
cat(rep("-", 60), "\n", sep = "")

set.seed(123)
t_start <- Sys.time()
result_hetero <- LOVE(X, pure_homo = FALSE, verbose = TRUE)
t_end <- Sys.time()

cat("\nResults (heterogeneous):\n")
cat("  Estimated K:", result_hetero$K, "\n")
cat("  Number of pure variables:", length(result_hetero$pureVec), "\n")
cat("  Pure variables:", paste(head(result_hetero$pureVec, 20), collapse = ", "),
    if(length(result_hetero$pureVec) > 20) ", ..." else "", "\n", sep = "")
cat("  optDelta:", result_hetero$optDelta, "\n")
cat("  Time:", format(t_end - t_start), "\n")

# Save outputs
write.csv(result_hetero$A, file.path(output_dir, "ssc_hetero_A.csv"), row.names = FALSE)
write.csv(result_hetero$C, file.path(output_dir, "ssc_hetero_C.csv"), row.names = FALSE)
write.csv(result_hetero$Gamma, file.path(output_dir, "ssc_hetero_Gamma.csv"), row.names = FALSE)
write.csv(data.frame(pureVec = result_hetero$pureVec),
          file.path(output_dir, "ssc_hetero_pureVec.csv"), row.names = FALSE)
write.csv(data.frame(K = result_hetero$K, optDelta = result_hetero$optDelta),
          file.path(output_dir, "ssc_hetero_params.csv"), row.names = FALSE)

# ============================================================================
# Run LOVE with homogeneous pure loadings
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Running LOVE (pure_homo = TRUE)\n")
cat(rep("-", 60), "\n", sep = "")

set.seed(123)
delta_grid <- seq(0.1, 1.1, 0.1)
t_start <- Sys.time()
result_homo <- LOVE(X, pure_homo = TRUE, delta = delta_grid, verbose = TRUE)
t_end <- Sys.time()

cat("\nResults (homogeneous):\n")
cat("  Estimated K:", result_homo$K, "\n")
cat("  Number of pure variables:", length(result_homo$pureVec), "\n")
cat("  Pure variables:", paste(head(result_homo$pureVec, 20), collapse = ", "),
    if(length(result_homo$pureVec) > 20) ", ..." else "", "\n", sep = "")
cat("  optDelta:", result_homo$optDelta, "\n")
cat("  Time:", format(t_end - t_start), "\n")

# Save outputs
write.csv(result_homo$A, file.path(output_dir, "ssc_homo_A.csv"), row.names = FALSE)
write.csv(result_homo$C, file.path(output_dir, "ssc_homo_C.csv"), row.names = FALSE)
write.csv(result_homo$Gamma, file.path(output_dir, "ssc_homo_Gamma.csv"), row.names = FALSE)
write.csv(data.frame(pureVec = result_homo$pureVec),
          file.path(output_dir, "ssc_homo_pureVec.csv"), row.names = FALSE)
write.csv(data.frame(K = result_homo$K, optDelta = result_homo$optDelta),
          file.path(output_dir, "ssc_homo_params.csv"), row.names = FALSE)

# ============================================================================
# Summary
# ============================================================================
cat("\n")
cat(rep("=", 60), "\n", sep = "")
cat("R outputs saved to:", output_dir, "\n")
cat("Now run: python run_ssc_comparison_py.py\n")
cat(rep("=", 60), "\n", sep = "")
