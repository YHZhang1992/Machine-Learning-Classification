full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", full_args, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "common.R"))

args <- parse_cli_args()
data_path <- arg_value(args, "data", required = TRUE)
label_col <- arg_value(args, "label", required = TRUE)
id_col <- arg_value(args, "id-column", default = NULL)
positive <- arg_value(args, "positive", default = NULL)
out_dir <- arg_value(args, "output", default = "results/q1_feature_optimization")
outer_folds <- as.integer(arg_value(args, "outer-folds", 5))
inner_folds <- as.integer(arg_value(args, "inner-folds", 4))
seed <- as.integer(arg_value(args, "seed", 42))

input <- load_xy(data_path, label_col, id_col, positive)
k_values <- candidate_feature_counts(ncol(input$x), parse_int_list(arg_value(args, "feature-counts", NULL)))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

res <- nested_cv_model(input$x, input$y, "ElasticNet", k_values, outer_folds, inner_folds, seed)
write.csv(res$fold_results, file.path(out_dir, "outer_fold_metrics.csv"), row.names = FALSE)
write.csv(summarize_outer_results(res$fold_results), file.path(out_dir, "performance_summary.csv"), row.names = FALSE)

all_selected <- unlist(res$selections, use.names = FALSE)
counts <- table(factor(all_selected, levels = colnames(input$x)))
stability <- data.frame(feature = names(counts), selection_count = as.integer(counts), selection_rate = as.numeric(counts) / length(res$selections), stringsAsFactors = FALSE)
stability <- stability[order(-stability$selection_rate, stability$feature), ]
write.csv(stability, file.path(out_dir, "feature_stability.csv"), row.names = FALSE)

final_k <- as.integer(median(res$fold_results$best_k))
final_features <- head(rank_features(input$x, input$y), final_k)
writeLines(final_features, file.path(out_dir, "candidate_feature_panel.txt"))
write_json(file.path(out_dir, "feature_optimization_metadata.json"), list(
  question = "Q1 - optimize predictive feature combination",
  reference_model = "ElasticNet",
  candidate_feature_counts = k_values,
  final_candidate_k = final_k,
  positive_label = input$positive,
  negative_label = input$negative,
  outer_folds = outer_folds,
  inner_folds = inner_folds,
  seed = seed,
  note = "Feature ranking and feature-count tuning are nested inside CV. The final candidate panel is for downstream freezing, not unbiased performance estimation.",
  outer_best_params = res$best_params
))
