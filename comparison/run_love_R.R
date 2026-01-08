#!/usr/bin/env Rscript
#
# R script to run LOVE on input data and save outputs for comparison
#
# Usage: Rscript run_love_R.R <data_file> [tag] [--fixed-delta VALUE]
#
# Arguments:
#   data_file       Path to CSV file (samples as rows, features as columns)
#   tag             Optional run name tag (default: basename of data file)
#   --fixed-delta   Use a fixed delta value to bypass CV (deterministic)
#
# Examples:
#   Rscript run_love_R.R /path/to/data.csv
#   Rscript run_love_R.R /path/to/data.csv my_experiment
#   Rscript run_love_R.R /path/to/data.csv my_experiment --fixed-delta 0.5
#

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  cat("Usage: Rscript run_love_R.R <data_file> [tag] [--fixed-delta VALUE]\n")
  cat("\nArguments:\n")
  cat("  data_file       Path to CSV file (samples as rows, features as columns)\n")
  cat("  tag             Optional run name tag (default: basename of data file)\n")
  cat("  --fixed-delta   Use a fixed delta value to bypass CV (deterministic)\n")
  cat("\nExamples:\n")
  cat("  Rscript run_love_R.R /path/to/data.csv my_experiment\n")
  cat("  Rscript run_love_R.R /path/to/data.csv my_experiment --fixed-delta 0.5\n")
  quit(status = 1)
}

data_file <- args[1]

# Check if data file exists
if (!file.exists(data_file)) {
  cat("ERROR: Data file not found:", data_file, "\n")
  quit(status = 1)
}

# Parse optional arguments
tag <- NULL
fixed_delta <- NULL

i <- 2
while (i <= length(args)) {
  if (args[i] == "--fixed-delta" && i < length(args)) {
    fixed_delta <- as.numeric(args[i + 1])
    i <- i + 2
  } else if (is.null(tag) && !startsWith(args[i], "--")) {
    tag <- args[i]
    i <- i + 1
  } else {
    i <- i + 1
  }
}

# Default tag to basename without extension
if (is.null(tag)) {
  tag <- tools::file_path_sans_ext(basename(data_file))
}

# Get script directory for sourcing R files
script_args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", script_args[grep("--file=", script_args)])
if (length(script_path) == 0) {
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

# Source all R files
r_dir <- file.path(script_dir, "..", "R")
r_files <- c("Utilities.R", "Score.R", "EstPureHomo.R", "EstPureHetero.R",
             "EstNonpure.R", "EstOmega.R", "PreScreen.R", "CV.R", "LOVE.R")

cat("Loading R source files from:", r_dir, "\n")
for (f in r_files) {
  fpath <- file.path(r_dir, f)
  if (!file.exists(fpath)) {
    cat("ERROR: R source file not found:", fpath, "\n")
    quit(status = 1)
  }
  source(fpath)
}

# Create output directory
output_dir <- file.path(script_dir, "outputs", tag)
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat(rep("=", 60), "\n", sep = "")
cat("LOVE R Analysis\n")
cat("  Data file: ", data_file, "\n", sep = "")
cat("  Tag: ", tag, "\n", sep = "")
cat("  Output dir: ", output_dir, "\n", sep = "")
if (!is.null(fixed_delta)) {
  cat("  Fixed delta: ", fixed_delta, " (CV bypassed for deterministic comparison)\n", sep = "")
}
cat(rep("=", 60), "\n", sep = "")

# Load data
cat("\nLoading data...\n")
X_df <- read.csv(data_file, row.names = 1)
X <- as.matrix(X_df)

n <- nrow(X)
p <- ncol(X)

cat("Data dimensions: ", n, " samples x ", p, " features\n", sep = "")

# Check dimensions
if (n <= p) {
  cat("\nWARNING: n (", n, ") <= p (", p, ")\n", sep = "")
  cat("LOVE requires more samples than features for reliable estimation.\n")
  cat("Consider transposing your data or using a subset of features.\n\n")
}

cat("Sample names: ", paste(head(rownames(X)), collapse = ", "),
    if (n > 6) ", ..." else "", "\n", sep = "")
cat("Feature names: ", paste(head(colnames(X), 5), collapse = ", "),
    if (p > 5) ", ..." else "", "\n", sep = "")

# Save the matrix for Python to use (ensures same data)
write.csv(X, file.path(output_dir, paste0(tag, "_X_matrix.csv")), row.names = TRUE)

# ============================================================================
# Run LOVE with heterogeneous pure loadings
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Running LOVE (pure_homo = FALSE)\n")
cat(rep("-", 60), "\n", sep = "")

set.seed(123)
t_start <- Sys.time()
result_hetero <- tryCatch({
  if (!is.null(fixed_delta)) {
    # Single delta value bypasses CV for deterministic comparison
    LOVE(X, pure_homo = FALSE, delta = fixed_delta, verbose = TRUE)
  } else {
    LOVE(X, pure_homo = FALSE, verbose = TRUE)
  }
}, error = function(e) {
  cat("ERROR in LOVE (hetero):", conditionMessage(e), "\n")
  return(NULL)
})
t_end <- Sys.time()

if (!is.null(result_hetero)) {
  cat("\nResults (heterogeneous):\n")
  cat("  Estimated K:", result_hetero$K, "\n")
  cat("  Number of pure variables:", length(result_hetero$pureVec), "\n")
  cat("  Pure variables:", paste(head(result_hetero$pureVec, 20), collapse = ", "),
      if(length(result_hetero$pureVec) > 20) ", ..." else "", "\n", sep = "")
  cat("  optDelta:", result_hetero$optDelta, "\n")
  cat("  Time:", format(t_end - t_start), "\n")

  # Save outputs
  write.csv(result_hetero$A, file.path(output_dir, paste0(tag, "_hetero_A.csv")), row.names = FALSE)
  write.csv(result_hetero$C, file.path(output_dir, paste0(tag, "_hetero_C.csv")), row.names = FALSE)
  write.csv(result_hetero$Gamma, file.path(output_dir, paste0(tag, "_hetero_Gamma.csv")), row.names = FALSE)
  write.csv(data.frame(pureVec = result_hetero$pureVec),
            file.path(output_dir, paste0(tag, "_hetero_pureVec.csv")), row.names = FALSE)
  write.csv(data.frame(K = result_hetero$K, optDelta = result_hetero$optDelta),
            file.path(output_dir, paste0(tag, "_hetero_params.csv")), row.names = FALSE)
} else {
  cat("Skipping heterogeneous outputs due to error.\n")
}

# ============================================================================
# Run LOVE with homogeneous pure loadings
# ============================================================================
cat("\n", rep("-", 60), "\n", sep = "")
cat("Running LOVE (pure_homo = TRUE)\n")
cat(rep("-", 60), "\n", sep = "")

set.seed(123)
t_start <- Sys.time()
result_homo <- tryCatch({
  if (!is.null(fixed_delta)) {
    # Single delta value bypasses CV for deterministic comparison
    LOVE(X, pure_homo = TRUE, delta = fixed_delta, verbose = TRUE)
  } else {
    delta_grid <- seq(0.1, 1.1, 0.1)
    LOVE(X, pure_homo = TRUE, delta = delta_grid, verbose = TRUE)
  }
}, error = function(e) {
  cat("ERROR in LOVE (homo):", conditionMessage(e), "\n")
  return(NULL)
})
t_end <- Sys.time()

if (!is.null(result_homo)) {
  cat("\nResults (homogeneous):\n")
  cat("  Estimated K:", result_homo$K, "\n")
  cat("  Number of pure variables:", length(result_homo$pureVec), "\n")
  cat("  Pure variables:", paste(head(result_homo$pureVec, 20), collapse = ", "),
      if(length(result_homo$pureVec) > 20) ", ..." else "", "\n", sep = "")
  cat("  optDelta:", result_homo$optDelta, "\n")
  cat("  Time:", format(t_end - t_start), "\n")

  # Save outputs
  write.csv(result_homo$A, file.path(output_dir, paste0(tag, "_homo_A.csv")), row.names = FALSE)
  write.csv(result_homo$C, file.path(output_dir, paste0(tag, "_homo_C.csv")), row.names = FALSE)
  write.csv(result_homo$Gamma, file.path(output_dir, paste0(tag, "_homo_Gamma.csv")), row.names = FALSE)
  write.csv(data.frame(pureVec = result_homo$pureVec),
            file.path(output_dir, paste0(tag, "_homo_pureVec.csv")), row.names = FALSE)
  write.csv(data.frame(K = result_homo$K, optDelta = result_homo$optDelta),
            file.path(output_dir, paste0(tag, "_homo_params.csv")), row.names = FALSE)
} else {
  cat("Skipping homogeneous outputs due to error.\n")
}

# ============================================================================
# Summary
# ============================================================================
cat("\n")
cat(rep("=", 60), "\n", sep = "")
cat("R outputs saved to:", output_dir, "\n")
cat("Now run: python run_love_py.py", data_file, tag, "\n")
cat(rep("=", 60), "\n", sep = "")
