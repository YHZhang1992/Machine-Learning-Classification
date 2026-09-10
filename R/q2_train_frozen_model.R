full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", full_args, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "common.R"))

args <- parse_cli_args()
data_path <- arg_value(args, "data", required = TRUE)
label_col <- arg_value(args, "label", required = TRUE)
id_col <- arg_value(args, "id-column", default = NULL)
positive <- arg_value(args, "positive", default = NULL)
out_dir <- arg_value(args, "output", default = "artifacts/R_frozen_model")
inner_folds <- as.integer(arg_value(args, "inner-folds", 5))
seed <- as.integer(arg_value(args, "seed", 42))
model_name <- arg_value(args, "model", default = NULL)
recommended_file <- arg_value(args, "recommended-model-file", default = NULL)
if (is.null(model_name) && !is.null(recommended_file)) model_name <- trimws(readLines(recommended_file, warn = FALSE)[[1]])
if (is.null(model_name)) stop("Provide either --model or --recommended-model-file.")
if (!model_name %in% MODEL_ORDER) stop("Unsupported model: ", model_name)

input <- load_xy(data_path, label_col, id_col, positive)
k_values <- candidate_feature_counts(ncol(input$x), parse_int_list(arg_value(args, "feature-counts", NULL)))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

artifact <- fit_frozen_model(input$x, input$y, model_name, k_values, inner_folds, seed)
artifact$positive_label <- input$positive
artifact$negative_label <- input$negative
artifact$training_feature_columns <- colnames(input$x)
artifact$seed <- seed
artifact$package_versions <- list(
  R = R.version.string,
  caret = as.character(utils::packageVersion("caret")),
  glmnet = as.character(utils::packageVersion("glmnet")),
  randomForest = as.character(utils::packageVersion("randomForest")),
  e1071 = as.character(utils::packageVersion("e1071")),
  pROC = as.character(utils::packageVersion("pROC"))
)
saveRDS(artifact, file.path(out_dir, "frozen_model.rds"))
writeLines(artifact$selected_features, file.path(out_dir, "selected_features.txt"))
write_json(file.path(out_dir, "frozen_model_metadata.json"), list(
  question = "Q2 - train a frozen model for transfer",
  model_name = model_name,
  selected_features = artifact$selected_features,
  positive_label = input$positive,
  negative_label = input$negative,
  best_search_params = artifact$best_search_params,
  inner_cv_roc_auc = artifact$inner_cv_roc_auc,
  seed = seed,
  package_versions = artifact$package_versions,
  important = "Performance should be reported from Q3 nested CV and/or an untouched external validation cohort, not from this full-development-data refit."
))
