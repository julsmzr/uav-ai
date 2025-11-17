run_full_evaluation_r <- function(metrics_txt_filepath, n_splits) {

  col_names <- c("V1", "Precision", "Recall", "mAP50", "mAP50_95")
  col_classes <- c("NULL", "numeric", "numeric", "numeric", "numeric")
  
  raw_data <- read.csv(metrics_txt_filepath, 
                       header = FALSE, 
                       sep = ",", 
                       col.names = col_names,
                       colClasses = col_classes)
  
  n_rows <- nrow(raw_data)
  
  expected_rows <- 375
  if (n_rows != expected_rows) {
    cat(sprintf("Warning: data is incomplete, expected: %d found: %d\n", expected_rows, n_rows))
  }
  
  # Create a grouping factor to average every n_splits rows
  group_id <- rep(1:ceiling(n_rows / n_splits), each = n_splits, length.out = n_rows)
  
  # Aggregate (average) by the grouping factor
  data_to_agg <- data.frame(raw_data, group_id = group_id)
  averaged_data <- aggregate(. ~ group_id, data = data_to_agg, FUN = mean)
  
  # The 'results' data frame now holds the 75 averaged experiment results, Remove the 'group_id' column, which is no longer needed
  results <- averaged_data[, -1] 
  
  experiment_vz <- results[1:25, ]
  experiment_ir <- results[26:50, ] 
  experiment_hy <- results[51:nrow(results), ]
  
  metric_names <- c("Precision", "Recall", "mAP50", "mAP50_95")
  
  for (metric_name in metric_names) {
    
    metric_vz <- experiment_vz[[metric_name]] # Use [[ ]] to extract as vector
    metric_ir <- experiment_ir[[metric_name]]
    metric_hy <- experiment_hy[[metric_name]]
    
    # --- Friedman Test ---
    # Bind data into a matrix: columns are groups (treatments), rows are blocks (subjects)
    data_matrix <- cbind(metric_vz, metric_ir, metric_hy)
    
    friedman_result <- friedman.test(data_matrix)
    
    cat(paste0("\n", metric_name, ":\n"))
    cat(sprintf("  Friedman test: χ² = %.4f, p = %.4f\n", 
                friedman_result$statistic, friedman_result$p.value))
    
    if (friedman_result$p.value < 0.05) {
      cat("  * Significant difference detected - consider post-hoc tests\n")
    }
    
    # --- Paired Wilcoxon Tests (Post-hoc) ---
    wilcox_vz_ir <- wilcox.test(metric_vz, metric_ir, paired = TRUE)
    wilcox_ir_hy <- wilcox.test(metric_ir, metric_hy, paired = TRUE)
    wilcox_vz_hy <- wilcox.test(metric_vz, metric_hy, paired = TRUE)
    
    cat(sprintf("    Paired Wilcoxon (VZ vs IR): V = %.1f, p = %.4f\n", 
                wilcox_vz_ir$statistic, wilcox_vz_ir$p.value))
    cat(sprintf("    Paired Wilcoxon (IR vs HY): V = %.1f, p = %.4f\n", 
                wilcox_ir_hy$statistic, wilcox_ir_hy$p.value))
    cat(sprintf("    Paired Wilcoxon (VZ vs HY): V = %.1f, p = %.4f\n", 
                wilcox_vz_hy$statistic, wilcox_vz_hy$p.value))
    
    # --- Multiple Test Correction (Hommel's) ---
    p_values <- c(wilcox_vz_ir$p.value, 
                  wilcox_ir_hy$p.value, 
                  wilcox_vz_hy$p.value)
                  
    adjusted_p_values <- p.adjust(p_values, method = "hommel")
    
    cat("    Adjusted p-values (Hommel's method):\n")
    cat(sprintf("      VZ vs IR (adj. p): %.4f\n", adjusted_p_values[1]))
    cat(sprintf("      IR vs HY (adj. p): %.4f\n", adjusted_p_values[2]))
    cat(sprintf("      VZ vs HY (adj. p): %.4f\n", adjusted_p_values[3]))
  }
}

run_full_evaluation_r(
    metrics_txt_filepath = "results/metrics_test.txt", 
    n_splits = 5
)
