full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", full_args, value = TRUE)
script_dir <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[[1]]))) else getwd()
source(file.path(script_dir, "common.R"))

args <- parse_cli_args()
model_path <- arg_value(args, "model", required = TRUE)
data_path <- arg_value(args, "data", required = TRUE)
id_col <- arg_value(args, "id-column", default = NULL)
out_path <- arg_value(args, "output", default = "transfer_predictions.csv")

artifact <- readRDS(model_path)
dat <- read.csv(data_path, check.names = FALSE, stringsAsFactors = FALSE)
missing <- setdiff(artifact$selected_features, colnames(dat))
if (length(missing)) stop("Transfer data are missing required features: ", paste(head(missing, 20), collapse = ", "))

x <- dat[, artifact$selected_features, drop = FALSE]
xp <- apply_preprocessor(x, artifact$preprocessor)
prob <- predict_model_prob(artifact$model_name, artifact$model, xp)
pred <- ifelse(prob >= 0.5, artifact$positive_label, artifact$negative_label)
out <- data.frame(predicted_probability = prob, predicted_class = pred, check.names = FALSE)
if (!is.null(id_col)) {
  if (!id_col %in% colnames(dat)) stop("ID column absent from transfer data: ", id_col)
  out <- cbind(dat[, id_col, drop = FALSE], out)
}
write.csv(out, out_path, row.names = FALSE)
