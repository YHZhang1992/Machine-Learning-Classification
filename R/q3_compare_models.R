full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", full_args, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "common.R"))

args <- parse_cli_args()
data_path <- arg_value(args, "data", required = TRUE)
label_col <- arg_value(args, "label", required = TRUE)
id_col <- arg_value(args, "id-column", default = NULL)
positive <- arg_value(args, "positive", default = NULL)
out_dir <- arg_value(args, "output", default = "results/q3_model_comparison")
outer_folds <- as.integer(arg_value(args, "outer-folds", 5))
inner_folds <- as.integer(arg_value(args, "inner-folds", 4))
seed <- as.integer(arg_value(args, "seed", 42))
models <- trimws(strsplit(arg_value(args, "models", paste(MODEL_ORDER, collapse = ",")), ",", fixed = TRUE)[[1]])
unknown <- setdiff(models, MODEL_ORDER)
if (length(unknown)) stop("Unknown model(s): ", paste(unknown, collapse = ", "))

input <- load_xy(data_path, label_col, id_col, positive)
k_values <- candidate_feature_counts(ncol(input$x), parse_int_list(arg_value(args, "feature-counts", NULL)))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

all_rows <- list(); all_params <- list()
for (m in models) {
  res <- nested_cv_model(input$x, input$y, m, k_values, outer_folds, inner_folds, seed)
  all_rows[[m]] <- res$fold_results
  all_params[[m]] <- res$best_params
}
fold_table <- do.call(rbind, all_rows); row.names(fold_table) <- NULL
summary_table <- summarize_outer_results(fold_table)
summary_table <- summary_table[order(-summary_table$mean_roc_auc), ]
recommended <- recommend_model(summary_table)

write.csv(fold_table, file.path(out_dir, "outer_fold_metrics.csv"), row.names = FALSE)
write.csv(summary_table, file.path(out_dir, "model_comparison_summary.csv"), row.names = FALSE)
writeLines(recommended, file.path(out_dir, "recommended_model.txt"))
write_json(file.path(out_dir, "model_comparison_metadata.json"), list(
  question = "Q3 - compare model families and select the most suitable model",
  primary_metric = "ROC AUC",
  selection_rule = paste0("one-standard-error rule on mean outer-fold ROC AUC, then simplicity preference: ", paste(MODEL_ORDER, collapse = " > ")),
  positive_label = input$positive,
  negative_label = input$negative,
  candidate_feature_counts = k_values,
  outer_folds = outer_folds,
  inner_folds = inner_folds,
  seed = seed,
  recommended_model = recommended,
  outer_best_params = all_params
))
