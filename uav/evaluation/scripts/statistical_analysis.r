options(warn = -1)# Suppress warnings

run_full_evaluation_r <- function(
  metrics_csv_filepath, n_splits, implementation_name = "r"
) {

  full_df <- read.csv(metrics_csv_filepath, header = TRUE)

  raw_data <- data.frame(
    Precision = full_df$precision,
    Recall    = full_df$recall,
    mAP50     = full_df$mAP50,
    mAP50_95  = full_df$mAP50.95
  )

  n_rows <- nrow(raw_data)

  group_id <- rep(
    1:ceiling(n_rows / n_splits),
    each = n_splits, length.out = n_rows
  )

  data_to_agg <- data.frame(raw_data, group_id = group_id)
  averaged_data <- aggregate(. ~ group_id, data = data_to_agg, FUN = mean)
  results <- averaged_data[, -1]

  # Split into groups (VZ, IR, HY)
  n_res <- nrow(results)
  limit <- floor(n_res / 3)

  experiment_vz <- results[1:limit, ]
  experiment_ir <- results[(limit + 1):(2 * limit), ]
  experiment_hy <- results[(2 * limit + 1):(3 * limit), ]

  metric_names <- c("Precision", "Recall", "mAP50", "mAP50_95")

  cat("implementation,measured_metric,friedman_p,wilcoxon_p_1v2,wilcoxon_p_2v3,wilcoxon_p_1v3,hommel_p_1v2,hommel_p_2v3,hommel_p_1v3\n") # nolint

  for (metric_name in metric_names) {

    metric_vz <- experiment_vz[[metric_name]]
    metric_ir <- experiment_ir[[metric_name]]
    metric_hy <- experiment_hy[[metric_name]]

    data_matrix <- cbind(metric_vz, metric_ir, metric_hy)
    friedman_result <- friedman.test(data_matrix)
    f_p <- friedman_result$p.value

    w_vz_ir <- wilcox.test(metric_vz, metric_ir, paired = TRUE, exact = TRUE, correct = FALSE) # nolint
    w_ir_hy <- wilcox.test(metric_ir, metric_hy, paired = TRUE, exact = TRUE, correct = FALSE) # nolint
    w_vz_hy <- wilcox.test(metric_vz, metric_hy, paired = TRUE, exact = TRUE, correct = FALSE) # nolint

    w_p_1v2 <- w_vz_ir$p.value
    w_p_2v3 <- w_ir_hy$p.value
    w_p_1v3 <- w_vz_hy$p.value

    raw_ps <- c(w_p_1v2, w_p_2v3, w_p_1v3)
    adj_ps <- p.adjust(raw_ps, method = "hommel")

    h_p_1v2 <- adj_ps[1]
    h_p_2v3 <- adj_ps[2]
    h_p_1v3 <- adj_ps[3]

    cat(sprintf(
      "%s,%s,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f,%.16f\n",
      implementation_name,
      metric_name,
      f_p,
      w_p_1v2, w_p_2v3, w_p_1v3,
      h_p_1v2, h_p_2v3, h_p_1v3
    ))
  }
}

run_full_evaluation_r(
  metrics_csv_filepath = "results/metrics.csv",
  n_splits = 5,
  implementation_name = "r"
)