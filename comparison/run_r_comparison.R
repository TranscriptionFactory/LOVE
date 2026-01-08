#!/usr/bin/env Rscript
#
# R script to generate comparison outputs for validating Python translation
# Run this first, then run compare_outputs.py
#
# Usage: cd /ocean/projects/cis240075p/arosen1/1_misc/LOVE/comparison
#        Rscript run_r_comparison.R
#

# Get the directory where this script is located
args <- commandArgs(trailingOnly = FALSE)
script_path <- sub("--file=", "", args[grep("--file=", args)])
if (length(script_path) == 0) {
  # Running interactively - assume we're in the comparison directory
  script_dir <- getwd()
} else {
  script_dir <- dirname(normalizePath(script_path))
}

# Set working directory to script location
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
output_dir <- file.path(script_dir, "outputs")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

cat(rep("=", 60), "\n", sep = "")
cat("LOVE R Comparison Script\n")
cat(rep("=", 60), "\n", sep = "")

# ============================================================================
# Test 1: Basic LOVE with heterogeneous pure loadings
# ============================================================================
cat("\nTest 1: LOVE (pure_homo = FALSE)\n")
cat(rep("-", 40), "\n", sep = "")

set.seed(42)
p <- 6
n <- 100
K <- 2

A_true <- rbind(c(1, 0), c(-1, 0), c(0, 1), c(0, 1), c(1/3, 2/3), c(1/2, -1/2))
Z <- matrix(rnorm(n * K, sd = sqrt(2)), n, K)
E <- matrix(rnorm(n * p), n, p)
X <- Z %*% t(A_true) + E

# Save input data
write.csv(X, file.path(output_dir, "test1_X.csv"), row.names = FALSE)
write.csv(A_true, file.path(output_dir, "test1_A_true.csv"), row.names = FALSE)

# Run LOVE
set.seed(123)  # Seed for CV randomness
result1 <- LOVE(X, pure_homo = FALSE, verbose = TRUE)

cat("Estimated K:", result1$K, "\n")
cat("Pure variables:", result1$pureVec, "\n")
cat("optDelta:", result1$optDelta, "\n")

# Save outputs
write.csv(result1$A, file.path(output_dir, "test1_A_hat.csv"), row.names = FALSE)
write.csv(result1$C, file.path(output_dir, "test1_C_hat.csv"), row.names = FALSE)
write.csv(result1$Gamma, file.path(output_dir, "test1_Gamma_hat.csv"), row.names = FALSE)
write.csv(data.frame(pureVec = result1$pureVec), file.path(output_dir, "test1_pureVec.csv"), row.names = FALSE)
write.csv(data.frame(K = result1$K, optDelta = result1$optDelta),
          file.path(output_dir, "test1_params.csv"), row.names = FALSE)

# ============================================================================
# Test 2: LOVE with homogeneous pure loadings
# ============================================================================
cat("\nTest 2: LOVE (pure_homo = TRUE)\n")
cat(rep("-", 40), "\n", sep = "")

# Regenerate data with same seed for consistency
set.seed(42)
Z <- matrix(rnorm(n * K, sd = sqrt(2)), n, K)
E <- matrix(rnorm(n * p), n, p)
X2 <- Z %*% t(A_true) + E

set.seed(123)
delta_grid <- seq(0.1, 1.1, 0.1)
result2 <- LOVE(X2, pure_homo = TRUE, delta = delta_grid, verbose = TRUE)

cat("Estimated K:", result2$K, "\n")
cat("Pure variables:", result2$pureVec, "\n")
cat("optDelta:", result2$optDelta, "\n")

# Save outputs
write.csv(result2$A, file.path(output_dir, "test2_A_hat.csv"), row.names = FALSE)
write.csv(result2$C, file.path(output_dir, "test2_C_hat.csv"), row.names = FALSE)
write.csv(result2$Gamma, file.path(output_dir, "test2_Gamma_hat.csv"), row.names = FALSE)
write.csv(data.frame(pureVec = result2$pureVec), file.path(output_dir, "test2_pureVec.csv"), row.names = FALSE)
write.csv(data.frame(K = result2$K, optDelta = result2$optDelta),
          file.path(output_dir, "test2_params.csv"), row.names = FALSE)

# ============================================================================
# Test 3: Screen_X pre-screening
# ============================================================================
cat("\nTest 3: Screen_X\n")
cat(rep("-", 40), "\n", sep = "")

set.seed(42)
# Add a pure noise feature
aug_A <- rbind(A_true, c(0, 0))
aug_p <- nrow(aug_A)
Z <- matrix(rnorm(n * K, sd = sqrt(2)), n, K)
E3 <- matrix(rnorm(n * aug_p), n, aug_p)
X3 <- Z %*% t(aug_A) + E3

write.csv(X3, file.path(output_dir, "test3_X.csv"), row.names = FALSE)

set.seed(123)
screen_result <- Screen_X(X3)

cat("Noise indices:", screen_result$noise_ind, "\n")
cat("Optimal threshold:", screen_result$thresh_min, "\n")

write.csv(data.frame(noise_ind = screen_result$noise_ind),
          file.path(output_dir, "test3_noise_ind.csv"), row.names = FALSE)
write.csv(data.frame(thresh_min = screen_result$thresh_min,
                     thresh_1se = screen_result$thresh_1se),
          file.path(output_dir, "test3_params.csv"), row.names = FALSE)

# ============================================================================
# Test 4: Score_mat computation
# ============================================================================
cat("\nTest 4: Score_mat\n")
cat(rep("-", 40), "\n", sep = "")

set.seed(42)
X4 <- matrix(rnorm(50 * 5), 50, 5)
R4 <- cor(X4)

score_result <- Score_mat(R4, q = 2, exact = FALSE)

write.csv(X4, file.path(output_dir, "test4_X.csv"), row.names = FALSE)
write.csv(R4, file.path(output_dir, "test4_R.csv"), row.names = FALSE)
write.csv(score_result$score, file.path(output_dir, "test4_score.csv"), row.names = FALSE)
write.csv(score_result$moments, file.path(output_dir, "test4_moments.csv"), row.names = FALSE)

cat("Score matrix computed and saved.\n")

# ============================================================================
# Test 5: EstC (covariance estimation)
# ============================================================================
cat("\nTest 5: EstC\n")
cat(rep("-", 40), "\n", sep = "")

# Regenerate X for consistency
set.seed(42)
Z <- matrix(rnorm(n * K, sd = sqrt(2)), n, K)
E <- matrix(rnorm(n * p), n, p)
X <- Z %*% t(A_true) + E

# Use a known loading matrix
AI_test <- matrix(0, 6, 2)
AI_test[1, 1] <- 1
AI_test[2, 1] <- -1
AI_test[3, 2] <- 1
AI_test[4, 2] <- 1

Sigma_test <- cov(X)

C_est <- EstC(Sigma_test, AI_test, diagonal = FALSE)
C_est_diag <- EstC(Sigma_test, AI_test, diagonal = TRUE)

write.csv(AI_test, file.path(output_dir, "test5_AI.csv"), row.names = FALSE)
write.csv(Sigma_test, file.path(output_dir, "test5_Sigma.csv"), row.names = FALSE)
write.csv(C_est, file.path(output_dir, "test5_C.csv"), row.names = FALSE)
write.csv(C_est_diag, file.path(output_dir, "test5_C_diag.csv"), row.names = FALSE)

cat("C matrix (non-diagonal):\n")
print(C_est)
cat("C matrix (diagonal):\n")
print(C_est_diag)

# ============================================================================
# Test 6: estOmega (precision matrix estimation)
# ============================================================================
cat("\nTest 6: estOmega\n")
cat(rep("-", 40), "\n", sep = "")

# Use the estimated C from test 1
if (!is.null(result1$C) && nrow(result1$C) > 0) {
  C_for_omega <- result1$C
  lbd_test <- 0.1

  Omega_est <- estOmega(lbd_test, C_for_omega)

  write.csv(C_for_omega, file.path(output_dir, "test6_C.csv"), row.names = FALSE)
  write.csv(Omega_est, file.path(output_dir, "test6_Omega.csv"), row.names = FALSE)
  write.csv(data.frame(lbd = lbd_test), file.path(output_dir, "test6_params.csv"), row.names = FALSE)

  cat("Omega matrix estimated and saved.\n")
  print(Omega_est)
} else {
  cat("Skipping estOmega test (no valid C matrix).\n")
}

# ============================================================================
# Summary
# ============================================================================
cat("\n")
cat(rep("=", 60), "\n", sep = "")
cat("R outputs saved to:", output_dir, "\n")
cat("Now run: python compare_outputs.py\n")
cat(rep("=", 60), "\n", sep = "")
