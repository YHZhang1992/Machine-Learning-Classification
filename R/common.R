MODEL_ORDER <- c("ElasticNet", "NaiveBayes", "SVM", "RandomForest")

parse_cli_args <- function(x = commandArgs(trailingOnly = TRUE)) {
  out <- list(); i <- 1L
  while (i <= length(x)) {
    token <- x[[i]]
    if (!startsWith(token, "--")) stop("Unexpected argument: ", token)
    token <- sub("^--", "", token)
    key <- token
    if (i < length(x) && !startsWith(x[[i + 1L]], "--")) {
      out[[key]] <- x[[i + 1L]]; i <- i + 2L
    } else { out[[key]] <- TRUE; i <- i + 1L }
  }
  out
}

arg_value <- function(args, key, default = NULL, required = FALSE) {
  value <- args[[key]]; if (is.null(value)) value <- default
  if (required && (is.null(value) || identical(value, ""))) stop("Missing required --", key)
  value
}

parse_int_list <- function(x) {
  if (is.null(x) || identical(x, "")) return(NULL)
  as.integer(strsplit(x, ",", fixed = TRUE)[[1]])
}

candidate_feature_counts <- function(p, requested = NULL) {
  vals <- if (is.null(requested)) c(5, 10, 20, 30, 50, 100, p) else requested
  vals <- sort(unique(as.integer(vals[vals > 0 & vals <= p])))
  sort(unique(c(vals, p)))
}

load_xy <- function(csv_path, label_column, id_column = NULL, positive_label = NULL) {
  dat <- read.csv(csv_path, check.names = FALSE, stringsAsFactors = FALSE)
  if (!label_column %in% colnames(dat)) stop("Label column not found: ", label_column)
  ids <- NULL
  if (!is.null(id_column)) {
    if (!id_column %in% colnames(dat)) stop("ID column not found: ", id_column)
    ids <- dat[[id_column]]
  }
  raw_y <- as.character(dat[[label_column]])
  levels_y <- sort(unique(raw_y[!is.na(raw_y)]))
  if (length(levels_y) != 2L) stop("Binary classification requires exactly 2 outcome levels")
  positive <- if (is.null(positive_label)) levels_y[[2]] else as.character(positive_label)
  if (!positive %in% levels_y) stop("Positive label not observed: ", positive)
  negative <- setdiff(levels_y, positive)[[1]]
  y <- as.integer(raw_y == positive)
  x <- dat[, setdiff(colnames(dat), c(label_column, id_column)), drop = FALSE]
  non_numeric <- colnames(x)[!vapply(x, is.numeric, logical(1))]
  if (length(non_numeric)) stop("All predictors must be numeric: ", paste(head(non_numeric, 20), collapse = ", "))
  list(x = x, y = y, positive = positive, negative = negative, ids = ids)
}

fit_medians <- function(x) vapply(x, function(z) { m <- median(z, na.rm = TRUE); if (!is.finite(m)) 0 else m }, numeric(1))

apply_medians <- function(x, medians) {
  out <- as.data.frame(x, check.names = FALSE)
  for (nm in colnames(out)) { z <- out[[nm]]; z[is.na(z)] <- medians[[nm]]; out[[nm]] <- z }
  out
}

rank_features <- function(x, y) {
  xi <- apply_medians(x, fit_medians(x))
  scores <- vapply(xi, function(z) suppressWarnings(abs(cor(z, y))), numeric(1))
  scores[!is.finite(scores)] <- -Inf
  names(sort(scores, decreasing = TRUE))
}

fit_preprocessor <- function(x) {
  med <- fit_medians(x); xi <- apply_medians(x, med)
  mu <- vapply(xi, mean, numeric(1)); sdv <- vapply(xi, sd, numeric(1)); sdv[!is.finite(sdv) | sdv == 0] <- 1
  list(medians = med, means = mu, sds = sdv)
}

apply_preprocessor <- function(x, pp) {
  mat <- as.matrix(apply_medians(x, pp$medians))
  mat <- sweep(mat, 2, pp$means[colnames(mat)], "-")
  mat <- sweep(mat, 2, pp$sds[colnames(mat)], "/")
  storage.mode(mat) <- "double"; mat
}

model_grid <- function(model_name) {
  if (model_name == "ElasticNet") return(expand.grid(alpha = c(0, .5, 1), lambda = c(.001, .01, .1, 1)))
  if (model_name == "RandomForest") return(expand.grid(mtry_frac = c(.25, .5), nodesize = c(1, 5)))
  if (model_name == "NaiveBayes") return(data.frame(laplace = 0))
  if (model_name == "SVM") return(expand.grid(cost = c(.1, 1, 10), gamma_factor = c(.25, 1, 4)))
  stop("Unknown model: ", model_name)
}

fit_model <- function(model_name, x, y, params, seed = 42L) {
  set.seed(seed); yf <- factor(y, levels = c(0, 1))
  if (model_name == "ElasticNet") return(glmnet::glmnet(x, y, family = "binomial", alpha = params$alpha, lambda = params$lambda, standardize = FALSE))
  if (model_name == "RandomForest") {
    mtry <- max(1L, min(ncol(x), as.integer(round(params$mtry_frac * ncol(x)))))
    return(randomForest::randomForest(x = x, y = yf, ntree = 500, mtry = mtry, nodesize = as.integer(params$nodesize)))
  }
  if (model_name == "NaiveBayes") return(e1071::naiveBayes(x = x, y = yf, laplace = params$laplace))
  if (model_name == "SVM") return(e1071::svm(x = x, y = yf, kernel = "radial", cost = params$cost, gamma = params$gamma_factor / max(1, ncol(x)), probability = TRUE, scale = FALSE))
}

predict_model_prob <- function(model_name, fit, x) {
  if (model_name == "ElasticNet") return(as.numeric(predict(fit, newx = x, type = "response")))
  if (model_name == "RandomForest") return(as.numeric(predict(fit, newdata = x, type = "prob")[, "1"]))
  if (model_name == "NaiveBayes") return(as.numeric(predict(fit, newdata = x, type = "raw")[, "1"]))
  if (model_name == "SVM") { pred <- predict(fit, newdata = x, probability = TRUE); return(as.numeric(attr(pred, "probabilities")[, "1"])) }
}

safe_auc <- function(y, prob) as.numeric(pROC::auc(pROC::roc(y, prob, levels = c(0, 1), direction = "<", quiet = TRUE)))

average_precision <- function(y, prob) {
  ord <- order(prob, decreasing = TRUE); yy <- y[ord]; positives <- sum(yy == 1)
  if (!positives) return(NA_real_)
  p <- cumsum(yy == 1) / seq_along(yy); mean(p[yy == 1])
}

classification_metrics <- function(y, prob) {
  pred <- as.integer(prob >= .5); tp <- sum(pred == 1 & y == 1); tn <- sum(pred == 0 & y == 0); fp <- sum(pred == 1 & y == 0); fn <- sum(pred == 0 & y == 1)
  sens <- if (tp + fn > 0) tp / (tp + fn) else NA_real_; spec <- if (tn + fp > 0) tn / (tn + fp) else NA_real_
  prec <- if (tp + fp > 0) tp / (tp + fp) else 0; f1 <- if (2 * tp + fp + fn > 0) 2 * tp / (2 * tp + fp + fn) else 0
  c(roc_auc = safe_auc(y, prob), pr_auc = average_precision(y, prob), accuracy = mean(pred == y), balanced_accuracy = mean(c(sens, spec), na.rm = TRUE), sensitivity = sens, specificity = spec, precision = prec, f1 = f1, brier = mean((prob - y)^2))
}

make_folds <- function(y, k, seed) { set.seed(seed); caret::createFolds(factor(y, levels = c(0, 1)), k = k, list = TRUE, returnTrain = FALSE) }

expand_search_grid <- function(model_name, k_values) do.call(rbind, lapply(k_values, function(k) { x <- model_grid(model_name); x$k <- k; x }))

inner_tune <- function(x, y, model_name, k_values, inner_splits = 4L, seed = 42L) {
  folds <- make_folds(y, inner_splits, seed); grid <- expand_search_grid(model_name, k_values); grid$mean_inner_roc_auc <- NA_real_
  for (g in seq_len(nrow(grid))) {
    scores <- numeric(length(folds))
    for (j in seq_along(folds)) {
      val <- folds[[j]]; tr <- setdiff(seq_len(nrow(x)), val); k <- min(as.integer(grid$k[g]), ncol(x)); feats <- head(rank_features(x[tr, , drop = FALSE], y[tr]), k)
      pp <- fit_preprocessor(x[tr, feats, drop = FALSE]); xtr <- apply_preprocessor(x[tr, feats, drop = FALSE], pp); xval <- apply_preprocessor(x[val, feats, drop = FALSE], pp)
      params <- grid[g, setdiff(colnames(grid), c("k", "mean_inner_roc_auc")), drop = FALSE]
      fit <- fit_model(model_name, xtr, y[tr], params, seed + g + j); scores[j] <- safe_auc(y[val], predict_model_prob(model_name, fit, xval))
    }
    grid$mean_inner_roc_auc[g] <- mean(scores, na.rm = TRUE)
  }
  grid[order(-grid$mean_inner_roc_auc, grid$k)[1], , drop = FALSE]
}

nested_cv_model <- function(x, y, model_name, k_values, outer_splits = 5L, inner_splits = 4L, seed = 42L) {
  outer <- make_folds(y, outer_splits, seed); rows <- list(); selections <- list(); params_out <- list()
  for (i in seq_along(outer)) {
    te <- outer[[i]]; tr <- setdiff(seq_len(nrow(x)), te); best <- inner_tune(x[tr, , drop = FALSE], y[tr], model_name, k_values, inner_splits, seed + 1000L + i)
    k <- min(as.integer(best$k), ncol(x)); feats <- head(rank_features(x[tr, , drop = FALSE], y[tr]), k); pp <- fit_preprocessor(x[tr, feats, drop = FALSE])
    xtr <- apply_preprocessor(x[tr, feats, drop = FALSE], pp); xte <- apply_preprocessor(x[te, feats, drop = FALSE], pp)
    p <- best[, setdiff(colnames(best), c("k", "mean_inner_roc_auc")), drop = FALSE]; fit <- fit_model(model_name, xtr, y[tr], p, seed + i); prob <- predict_model_prob(model_name, fit, xte)
    rows[[i]] <- data.frame(model = model_name, outer_fold = i, inner_best_roc_auc = best$mean_inner_roc_auc, best_k = k, as.list(classification_metrics(y[te], prob)), check.names = FALSE)
    selections[[i]] <- feats; params_out[[i]] <- as.list(best)
  }
  list(fold_results = do.call(rbind, rows), selections = selections, best_params = params_out)
}

summarize_outer_results <- function(d) {
  metrics <- c("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "f1", "brier")
  do.call(rbind, lapply(unique(d$model), function(m) {
    x <- d[d$model == m, , drop = FALSE]; row <- list(model = m, n_outer_folds = nrow(x))
    for (metric in metrics) { vals <- as.numeric(x[[metric]]); row[[paste0("mean_", metric)]] <- mean(vals, na.rm = TRUE); row[[paste0("sd_", metric)]] <- sd(vals, na.rm = TRUE); row[[paste0("se_", metric)]] <- sd(vals, na.rm = TRUE) / sqrt(sum(is.finite(vals))) }
    row$median_best_k <- as.integer(median(x$best_k)); as.data.frame(row, check.names = FALSE)
  }))
}

recommend_model <- function(summary_table) {
  best <- which.max(summary_table$mean_roc_auc); cutoff <- summary_table$mean_roc_auc[best] - summary_table$se_roc_auc[best]; eligible <- summary_table$model[summary_table$mean_roc_auc >= cutoff]
  for (m in MODEL_ORDER) if (m %in% eligible) return(m); summary_table$model[best]
}

fit_frozen_model <- function(x, y, model_name, k_values, inner_splits = 5L, seed = 42L) {
  best <- inner_tune(x, y, model_name, k_values, inner_splits, seed); k <- min(as.integer(best$k), ncol(x)); feats <- head(rank_features(x, y), k); pp <- fit_preprocessor(x[, feats, drop = FALSE]); xp <- apply_preprocessor(x[, feats, drop = FALSE], pp)
  p <- best[, setdiff(colnames(best), c("k", "mean_inner_roc_auc")), drop = FALSE]; fit <- fit_model(model_name, xp, y, p, seed)
  list(model = fit, model_name = model_name, selected_features = feats, preprocessor = pp, best_search_params = as.list(best), inner_cv_roc_auc = best$mean_inner_roc_auc)
}

write_json <- function(path, payload) jsonlite::write_json(payload, path, pretty = TRUE, auto_unbox = TRUE, null = "null")
