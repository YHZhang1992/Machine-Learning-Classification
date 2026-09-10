# ==============================================================================
# Machine Learning Classification Utility Functions
#
# Purpose:
#   Utility functions supporting binary classification using:
#     - Elastic Net
#     - Naive Bayes
#     - Support Vector Machine (radial kernel)
#     - Random Forest
#
# Major workflow components:
#   1. Command-line argument parsing
#   2. Input data loading and validation
#   3. Missing-value imputation
#   4. Feature ranking
#   5. Data preprocessing
#   6. Hyperparameter-grid construction
#   7. Model fitting and prediction
#   8. Classification performance evaluation
#   9. Inner cross-validation for tuning
#  10. Nested cross-validation for unbiased evaluation
#  11. Model comparison and recommendation
#  12. Final frozen-model fitting
#  13. JSON output
#
# Notes:
#   - Binary outcomes are internally encoded as 0/1.
#   - Feature ranking is performed using absolute Pearson correlation.
#   - Missing predictor values are median-imputed.
#   - Predictors are centered and scaled after imputation.
#   - Feature selection and preprocessing are learned only from training data
#     within each CV split to minimize information leakage.
# ==============================================================================


# ------------------------------------------------------------------------------
# Preferred model ordering
# ------------------------------------------------------------------------------

# Model preference order used when multiple models have statistically similar
# outer-cross-validation ROC-AUC performance.
#
# The order acts as a deterministic tie-breaking rule in recommend_model().
MODEL_ORDER <- c(
  "ElasticNet",
  "NaiveBayes",
  "SVM",
  "RandomForest"
)


# ==============================================================================
# Command-line argument utilities
# ==============================================================================


#' Parse command-line arguments
#'
#' Parses command-line arguments supplied using GNU-style double-dash syntax.
#'
#' Supported formats include:
#'
#'   --input data.csv
#'   --label outcome
#'   --seed 42
#'   --flag
#'
#' Arguments followed by a value are stored as character values. Arguments
#' without an accompanying value are interpreted as logical flags and assigned
#' TRUE.
#'
#' @param x Character vector containing command-line arguments. By default,
#'   arguments are obtained using `commandArgs(trailingOnly = TRUE)`.
#'
#' @return A named list in which each element corresponds to one command-line
#'   argument. Flag-only arguments are represented by TRUE.
#'
#' @details
#' Every argument must begin with `"--"`. An error is raised when an unexpected
#' positional argument is encountered.
#'
#' @examples
#' parse_cli_args(c("--input", "data.csv", "--seed", "42", "--verbose"))
parse_cli_args <- function(x = commandArgs(trailingOnly = TRUE)) {

  out <- list()
  i <- 1L

  while (i <= length(x)) {

    token <- x[[i]]

    if (!startsWith(token, "--")) {
      stop("Unexpected argument: ", token)
    }

    token <- sub("^--", "", token)
    key <- token

    if (
      i < length(x) &&
      !startsWith(x[[i + 1L]], "--")
    ) {

      out[[key]] <- x[[i + 1L]]
      i <- i + 2L

    } else {

      out[[key]] <- TRUE
      i <- i + 1L
    }
  }

  out
}


#' Retrieve one parsed command-line argument
#'
#' Extracts a named argument from the output of `parse_cli_args()`.
#'
#' @param args Named list of parsed command-line arguments.
#' @param key Character scalar giving the argument name without the leading
#'   `"--"`.
#' @param default Default value returned when the requested argument is absent.
#'   Defaults to NULL.
#' @param required Logical indicating whether the argument must be provided.
#'   Defaults to FALSE.
#'
#' @return The requested argument value, or `default` when the argument is
#'   absent.
#'
#' @details
#' If `required = TRUE`, an error is raised when the argument is missing or is
#' an empty string.
#'
#' @examples
#' args <- list(input = "data.csv")
#' arg_value(args, "input", required = TRUE)
arg_value <- function(
  args,
  key,
  default = NULL,
  required = FALSE
) {

  value <- args[[key]]

  if (is.null(value)) {
    value <- default
  }

  if (
    required &&
    (is.null(value) || identical(value, ""))
  ) {
    stop("Missing required --", key)
  }

  value
}


#' Parse a comma-separated integer list
#'
#' Converts a character string containing comma-separated integer values into
#' an integer vector.
#'
#' @param x Character scalar such as `"5,10,20,50"`. May also be NULL or an
#'   empty string.
#'
#' @return Integer vector containing the parsed values. Returns NULL if `x` is
#'   NULL or empty.
#'
#' @examples
#' parse_int_list("5,10,20,50")
parse_int_list <- function(x) {

  if (
    is.null(x) ||
    identical(x, "")
  ) {
    return(NULL)
  }

  as.integer(
    strsplit(
      x,
      ",",
      fixed = TRUE
    )[[1]]
  )
}


# ==============================================================================
# Feature-count configuration
# ==============================================================================


#' Generate candidate feature-set sizes
#'
#' Creates the candidate numbers of top-ranked predictors to evaluate during
#' model tuning.
#'
#' @param p Integer giving the total number of available predictor variables.
#' @param requested Optional integer vector containing user-requested feature
#'   counts. When NULL, the default candidate values are
#'   `5, 10, 20, 30, 50, 100, p`.
#'
#' @return Sorted integer vector of unique candidate feature counts.
#'
#' @details
#' Values less than or equal to zero and values larger than the total number of
#' predictors are removed. The complete predictor set (`p`) is always included
#' as a candidate.
#'
#' @examples
#' candidate_feature_counts(75)
#' candidate_feature_counts(75, c(10, 25, 50))
candidate_feature_counts <- function(
  p,
  requested = NULL
) {

  vals <- if (is.null(requested)) {
    c(5, 10, 20, 30, 50, 100, p)
  } else {
    requested
  }

  vals <- sort(
    unique(
      as.integer(
        vals[
          vals > 0 &
          vals <= p
        ]
      )
    )
  )

  sort(
    unique(
      c(vals, p)
    )
  )
}


# ==============================================================================
# Data loading and validation
# ==============================================================================


#' Load predictors and binary outcome from a CSV file
#'
#' Reads a CSV dataset, validates the requested label and optional identifier
#' columns, converts the binary outcome into 0/1 encoding, and extracts numeric
#' predictors.
#'
#' @param csv_path Character path to the input CSV file.
#' @param label_column Character name of the binary outcome column.
#' @param id_column Optional character name of a sample or subject identifier
#'   column. Defaults to NULL.
#' @param positive_label Optional label defining the positive outcome class.
#'   When NULL, the second alphabetically sorted outcome level is used as the
#'   positive class.
#'
#' @return A list containing:
#'
#'   \itemize{
#'     \item `x`: Predictor data frame.
#'     \item `y`: Binary integer outcome vector encoded as 0/1.
#'     \item `positive`: Original label representing class 1.
#'     \item `negative`: Original label representing class 0.
#'     \item `ids`: Identifier vector when `id_column` is supplied; otherwise
#'       NULL.
#'   }
#'
#' @details
#' Exactly two observed outcome levels are required. All predictor columns must
#' be numeric. The label and optional ID columns are excluded from the predictor
#' matrix.
#'
#' @examples
#' dat <- load_xy(
#'   csv_path = "training.csv",
#'   label_column = "response",
#'   id_column = "sample_id",
#'   positive_label = "Responder"
#' )
load_xy <- function(
  csv_path,
  label_column,
  id_column = NULL,
  positive_label = NULL
) {

  dat <- read.csv(
    csv_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  if (!label_column %in% colnames(dat)) {
    stop(
      "Label column not found: ",
      label_column
    )
  }

  ids <- NULL

  if (!is.null(id_column)) {

    if (!id_column %in% colnames(dat)) {
      stop(
        "ID column not found: ",
        id_column
      )
    }

    ids <- dat[[id_column]]
  }

  raw_y <- as.character(
    dat[[label_column]]
  )

  levels_y <- sort(
    unique(
      raw_y[
        !is.na(raw_y)
      ]
    )
  )

  if (length(levels_y) != 2L) {
    stop(
      "Binary classification requires exactly 2 outcome levels"
    )
  }

  positive <- if (is.null(positive_label)) {
    levels_y[[2]]
  } else {
    as.character(positive_label)
  }

  if (!positive %in% levels_y) {
    stop(
      "Positive label not observed: ",
      positive
    )
  }

  negative <- setdiff(
    levels_y,
    positive
  )[[1]]

  y <- as.integer(
    raw_y == positive
  )

  x <- dat[
    ,
    setdiff(
      colnames(dat),
      c(
        label_column,
        id_column
      )
    ),
    drop = FALSE
  ]

  non_numeric <- colnames(x)[
    !vapply(
      x,
      is.numeric,
      logical(1)
    )
  ]

  if (length(non_numeric)) {
    stop(
      "All predictors must be numeric: ",
      paste(
        head(non_numeric, 20),
        collapse = ", "
      )
    )
  }

  list(
    x = x,
    y = y,
    positive = positive,
    negative = negative,
    ids = ids
  )
}


# ==============================================================================
# Missing-value imputation
# ==============================================================================


#' Estimate predictor-specific medians
#'
#' Calculates the median value of each predictor for later missing-value
#' imputation.
#'
#' @param x Data frame or list-like object containing numeric predictors.
#'
#' @return Named numeric vector containing one median per predictor.
#'
#' @details
#' Missing values are ignored when calculating medians. If a predictor does not
#' have a finite median, its imputation value is set to zero.
fit_medians <- function(x) {

  vapply(
    x,
    function(z) {

      m <- median(
        z,
        na.rm = TRUE
      )

      if (!is.finite(m)) {
        0
      } else {
        m
      }
    },
    numeric(1)
  )
}


#' Apply median imputation
#'
#' Replaces missing predictor values using a previously estimated set of
#' feature-specific medians.
#'
#' @param x Predictor data frame.
#' @param medians Named numeric vector containing the imputation median for each
#'   predictor.
#'
#' @return Data frame with missing predictor values replaced by their
#'   corresponding medians.
apply_medians <- function(
  x,
  medians
) {

  out <- as.data.frame(
    x,
    check.names = FALSE
  )

  for (nm in colnames(out)) {

    z <- out[[nm]]

    z[
      is.na(z)
    ] <- medians[[nm]]

    out[[nm]] <- z
  }

  out
}


# ==============================================================================
# Feature ranking
# ==============================================================================


#' Rank predictors by univariate association with the outcome
#'
#' Ranks predictors according to the absolute Pearson correlation between each
#' predictor and the binary 0/1 outcome.
#'
#' @param x Data frame containing numeric predictors.
#' @param y Binary numeric outcome vector encoded as 0/1.
#'
#' @return Character vector containing predictor names ordered from strongest
#'   to weakest absolute correlation with the outcome.
#'
#' @details
#' Missing predictor values are first median-imputed. Absolute correlation is
#' used so that both positive and negative associations contribute equally to
#' feature ranking.
#'
#' Non-finite correlations, such as those produced by constant predictors, are
#' assigned `-Inf` and therefore ranked last.
#'
#' During nested cross-validation this function is applied only to the relevant
#' training partition so feature ranking is not learned from validation or test
#' observations.
rank_features <- function(
  x,
  y
) {

  xi <- apply_medians(
    x,
    fit_medians(x)
  )

  scores <- vapply(
    xi,
    function(z) {
      suppressWarnings(
        abs(
          cor(
            z,
            y
          )
        )
      )
    },
    numeric(1)
  )

  scores[
    !is.finite(scores)
  ] <- -Inf

  names(
    sort(
      scores,
      decreasing = TRUE
    )
  )
}


# ==============================================================================
# Predictor preprocessing
# ==============================================================================


#' Fit predictor preprocessing parameters
#'
#' Learns missing-value imputation, centering, and scaling parameters from a
#' training dataset.
#'
#' @param x Data frame containing numeric predictors.
#'
#' @return A list containing:
#'
#'   \itemize{
#'     \item `medians`: Feature-specific median imputation values.
#'     \item `means`: Feature means after median imputation.
#'     \item `sds`: Feature standard deviations after median imputation.
#'   }
#'
#' @details
#' Predictors with zero or non-finite standard deviation are assigned a scaling
#' value of 1 to avoid division-by-zero errors.
#'
#' The returned parameters should be learned from training data and reused
#' unchanged when preprocessing validation, test, or external datasets.
fit_preprocessor <- function(x) {

  med <- fit_medians(x)

  xi <- apply_medians(
    x,
    med
  )

  mu <- vapply(
    xi,
    mean,
    numeric(1)
  )

  sdv <- vapply(
    xi,
    sd,
    numeric(1)
  )

  sdv[
    !is.finite(sdv) |
    sdv == 0
  ] <- 1

  list(
    medians = med,
    means = mu,
    sds = sdv
  )
}


#' Apply fitted predictor preprocessing
#'
#' Applies median imputation, mean centering, and standard-deviation scaling
#' using parameters learned by `fit_preprocessor()`.
#'
#' @param x Predictor data frame.
#' @param pp Preprocessing object returned by `fit_preprocessor()`.
#'
#' @return Double-precision numeric matrix containing the transformed
#'   predictors.
#'
#' @details
#' No preprocessing parameters are recalculated in this function. This allows
#' the training-derived transformation to be applied consistently to validation,
#' test, and external datasets.
apply_preprocessor <- function(
  x,
  pp
) {

  mat <- as.matrix(
    apply_medians(
      x,
      pp$medians
    )
  )

  mat <- sweep(
    mat,
    2,
    pp$means[
      colnames(mat)
    ],
    "-"
  )

  mat <- sweep(
    mat,
    2,
    pp$sds[
      colnames(mat)
    ],
    "/"
  )

  storage.mode(mat) <- "double"

  mat
}


# ==============================================================================
# Hyperparameter grids
# ==============================================================================


#' Construct the hyperparameter grid for a classification algorithm
#'
#' Defines the candidate hyperparameter combinations evaluated during inner
#' cross-validation.
#'
#' @param model_name Character scalar specifying one of:
#'
#'   \itemize{
#'     \item `"ElasticNet"`
#'     \item `"RandomForest"`
#'     \item `"NaiveBayes"`
#'     \item `"SVM"`
#'   }
#'
#' @return Data frame containing candidate hyperparameter combinations for the
#'   specified model.
#'
#' @details
#' The grids are:
#'
#' Elastic Net:
#'   - alpha: 0, 0.5, 1
#'   - lambda: 0.001, 0.01, 0.1, 1
#'
#' Random Forest:
#'   - mtry fraction: 0.25, 0.5
#'   - node size: 1, 5
#'
#' Naive Bayes:
#'   - Laplace smoothing: 0
#'
#' SVM:
#'   - cost: 0.1, 1, 10
#'   - gamma factor: 0.25, 1, 4
#'
#' An error is raised for unsupported model names.
model_grid <- function(model_name) {

  if (model_name == "ElasticNet") {

    return(
      expand.grid(
        alpha = c(
          0,
          0.5,
          1
        ),
        lambda = c(
          0.001,
          0.01,
          0.1,
          1
        )
      )
    )
  }

  if (model_name == "RandomForest") {

    return(
      expand.grid(
        mtry_frac = c(
          0.25,
          0.5
        ),
        nodesize = c(
          1,
          5
        )
      )
    )
  }

  if (model_name == "NaiveBayes") {

    return(
      data.frame(
        laplace = 0
      )
    )
  }

  if (model_name == "SVM") {

    return(
      expand.grid(
        cost = c(
          0.1,
          1,
          10
        ),
        gamma_factor = c(
          0.25,
          1,
          4
        )
      )
    )
  }

  stop(
    "Unknown model: ",
    model_name
  )
}


# ==============================================================================
# Model fitting
# ==============================================================================


#' Fit a binary classification model
#'
#' Fits one of the supported machine-learning algorithms using a specified
#' hyperparameter configuration.
#'
#' @param model_name Character scalar specifying the algorithm:
#'   `"ElasticNet"`, `"RandomForest"`, `"NaiveBayes"`, or `"SVM"`.
#' @param x Numeric predictor matrix.
#' @param y Binary outcome vector encoded as 0/1.
#' @param params Data frame or list-like object containing the model-specific
#'   hyperparameters.
#' @param seed Integer random seed used to improve reproducibility.
#'
#' @return Fitted model object produced by the corresponding modeling package.
#'
#' @details
#' Implementations:
#'
#'   \itemize{
#'     \item Elastic Net: `glmnet::glmnet()`
#'     \item Random Forest: `randomForest::randomForest()`
#'     \item Naive Bayes: `e1071::naiveBayes()`
#'     \item SVM: `e1071::svm()` with radial kernel
#'   }
#'
#' Predictors are assumed to have already been processed by
#' `apply_preprocessor()`.
#'
#' Internal model scaling is disabled for Elastic Net and SVM because scaling
#' is handled by the workflow preprocessing functions.
fit_model <- function(
  model_name,
  x,
  y,
  params,
  seed = 42L
) {

  set.seed(seed)

  yf <- factor(
    y,
    levels = c(
      0,
      1
    )
  )

  if (model_name == "ElasticNet") {

    return(
      glmnet::glmnet(
        x,
        y,
        family = "binomial",
        alpha = params$alpha,
        lambda = params$lambda,
        standardize = FALSE
      )
    )
  }

  if (model_name == "RandomForest") {

    mtry <- max(
      1L,
      min(
        ncol(x),
        as.integer(
          round(
            params$mtry_frac *
            ncol(x)
          )
        )
      )
    )

    return(
      randomForest::randomForest(
        x = x,
        y = yf,
        ntree = 500,
        mtry = mtry,
        nodesize = as.integer(
          params$nodesize
        )
      )
    )
  }

  if (model_name == "NaiveBayes") {

    return(
      e1071::naiveBayes(
        x = x,
        y = yf,
        laplace = params$laplace
      )
    )
  }

  if (model_name == "SVM") {

    return(
      e1071::svm(
        x = x,
        y = yf,
        kernel = "radial",
        cost = params$cost,
        gamma = params$gamma_factor /
          max(
            1,
            ncol(x)
          ),
        probability = TRUE,
        scale = FALSE
      )
    )
  }
}


# ==============================================================================
# Model prediction
# ==============================================================================


#' Generate positive-class probabilities
#'
#' Produces predicted probabilities for class 1 from a fitted classification
#' model.
#'
#' @param model_name Character name of the fitted model type.
#' @param fit Fitted model object returned by `fit_model()`.
#' @param x Preprocessed predictor matrix for observations to score.
#'
#' @return Numeric vector containing the predicted probability of class 1.
#'
#' @details
#' Prediction syntax differs across the supported modeling packages. This
#' function provides a common interface and extracts the class-1 probability
#' consistently.
predict_model_prob <- function(
  model_name,
  fit,
  x
) {

  if (model_name == "ElasticNet") {

    return(
      as.numeric(
        predict(
          fit,
          newx = x,
          type = "response"
        )
      )
    )
  }

  if (model_name == "RandomForest") {

    return(
      as.numeric(
        predict(
          fit,
          newdata = x,
          type = "prob"
        )[
          ,
          "1"
        ]
      )
    )
  }

  if (model_name == "NaiveBayes") {

    return(
      as.numeric(
        predict(
          fit,
          newdata = x,
          type = "raw"
        )[
          ,
          "1"
        ]
      )
    )
  }

  if (model_name == "SVM") {

    pred <- predict(
      fit,
      newdata = x,
      probability = TRUE
    )

    return(
      as.numeric(
        attr(
          pred,
          "probabilities"
        )[
          ,
          "1"
        ]
      )
    )
  }
}


# ==============================================================================
# Classification performance metrics
# ==============================================================================


#' Calculate ROC AUC
#'
#' Computes the area under the receiver-operating-characteristic curve for
#' binary predictions.
#'
#' @param y Binary observed outcome vector encoded as 0/1.
#' @param prob Numeric vector containing predicted probabilities for class 1.
#'
#' @return Numeric ROC-AUC value.
#'
#' @details
#' ROC-AUC is calculated using the `pROC` package with class levels explicitly
#' defined as 0 and 1.
safe_auc <- function(
  y,
  prob
) {

  as.numeric(
    pROC::auc(
      pROC::roc(
        y,
        prob,
        levels = c(
          0,
          1
        ),
        direction = "<",
        quiet = TRUE
      )
    )
  )
}


#' Calculate average precision
#'
#' Computes average precision from ranked predicted probabilities.
#'
#' @param y Binary observed outcome vector encoded as 0/1.
#' @param prob Numeric vector containing predicted probabilities for class 1.
#'
#' @return Numeric average-precision value. Returns `NA_real_` when the observed
#'   data contain no positive samples.
#'
#' @details
#' Samples are ranked from highest to lowest predicted probability. Precision is
#' calculated at each rank, and the mean precision at ranks corresponding to
#' observed positive samples is returned.
average_precision <- function(
  y,
  prob
) {

  ord <- order(
    prob,
    decreasing = TRUE
  )

  yy <- y[ord]

  positives <- sum(
    yy == 1
  )

  if (!positives) {
    return(NA_real_)
  }

  p <- cumsum(
    yy == 1
  ) /
    seq_along(yy)

  mean(
    p[
      yy == 1
    ]
  )
}


#' Calculate binary classification performance metrics
#'
#' Evaluates predicted class probabilities using discrimination,
#' classification, and calibration metrics.
#'
#' @param y Binary observed outcome vector encoded as 0/1.
#' @param prob Numeric predicted probability vector for class 1.
#'
#' @return Named numeric vector containing:
#'
#'   \itemize{
#'     \item `roc_auc`: Receiver-operating-characteristic AUC.
#'     \item `pr_auc`: Average precision.
#'     \item `accuracy`: Overall classification accuracy.
#'     \item `balanced_accuracy`: Mean of sensitivity and specificity.
#'     \item `sensitivity`: True-positive rate.
#'     \item `specificity`: True-negative rate.
#'     \item `precision`: Positive predictive value.
#'     \item `f1`: F1 score.
#'     \item `brier`: Mean squared probability error.
#'   }
#'
#' @details
#' Probabilities greater than or equal to 0.5 are classified as positive.
classification_metrics <- function(
  y,
  prob
) {

  pred <- as.integer(
    prob >= 0.5
  )

  tp <- sum(
    pred == 1 &
    y == 1
  )

  tn <- sum(
    pred == 0 &
    y == 0
  )

  fp <- sum(
    pred == 1 &
    y == 0
  )

  fn <- sum(
    pred == 0 &
    y == 1
  )

  sens <- if (tp + fn > 0) {
    tp / (tp + fn)
  } else {
    NA_real_
  }

  spec <- if (tn + fp > 0) {
    tn / (tn + fp)
  } else {
    NA_real_
  }

  prec <- if (tp + fp > 0) {
    tp / (tp + fp)
  } else {
    0
  }

  f1 <- if (
    2 * tp +
    fp +
    fn > 0
  ) {

    2 * tp /
      (
        2 * tp +
        fp +
        fn
      )

  } else {

    0
  }

  c(
    roc_auc = safe_auc(
      y,
      prob
    ),

    pr_auc = average_precision(
      y,
      prob
    ),

    accuracy = mean(
      pred == y
    ),

    balanced_accuracy = mean(
      c(
        sens,
        spec
      ),
      na.rm = TRUE
    ),

    sensitivity = sens,

    specificity = spec,

    precision = prec,

    f1 = f1,

    brier = mean(
      (
        prob -
        y
      )^2
    )
  )
}


# ==============================================================================
# Cross-validation utilities
# ==============================================================================


#' Generate stratified cross-validation folds
#'
#' Creates approximately class-balanced validation folds for binary outcomes.
#'
#' @param y Binary outcome vector encoded as 0/1.
#' @param k Integer number of cross-validation folds.
#' @param seed Integer random seed.
#'
#' @return List of integer vectors. Each element contains the row indices used
#'   as the validation/test portion of one fold.
#'
#' @details
#' Fold creation is performed using `caret::createFolds()`. The binary outcome
#' is converted to a factor so fold construction is stratified by class.
make_folds <- function(
  y,
  k,
  seed
) {

  set.seed(seed)

  caret::createFolds(
    factor(
      y,
      levels = c(
        0,
        1
      )
    ),
    k = k,
    list = TRUE,
    returnTrain = FALSE
  )
}


#' Expand model hyperparameters across candidate feature counts
#'
#' Combines a model-specific hyperparameter grid with all candidate values of
#' `k`, where `k` represents the number of top-ranked features included in a
#' model.
#'
#' @param model_name Character model name accepted by `model_grid()`.
#' @param k_values Integer vector containing candidate feature-set sizes.
#'
#' @return Data frame containing every hyperparameter-by-feature-count
#'   combination.
expand_search_grid <- function(
  model_name,
  k_values
) {

  do.call(
    rbind,
    lapply(
      k_values,
      function(k) {

        x <- model_grid(
          model_name
        )

        x$k <- k

        x
      }
    )
  )
}


# ==============================================================================
# Inner cross-validation
# ==============================================================================


#' Tune feature count and model hyperparameters using inner cross-validation
#'
#' Performs the inner model-selection loop of a nested cross-validation
#' procedure.
#'
#' @param x Predictor data frame.
#' @param y Binary outcome vector encoded as 0/1.
#' @param model_name Character name of the machine-learning algorithm.
#' @param k_values Integer vector containing candidate numbers of selected
#'   features.
#' @param inner_splits Integer number of inner cross-validation folds.
#'   Defaults to 4.
#' @param seed Integer random seed.
#'
#' @return One-row data frame containing the best-performing combination of:
#'
#'   \itemize{
#'     \item Model hyperparameters
#'     \item Number of selected features (`k`)
#'     \item Mean inner-fold ROC AUC (`mean_inner_roc_auc`)
#'   }
#'
#' @details
#' For each candidate tuning configuration:
#'
#'   1. Stratified inner folds are generated.
#'   2. Feature ranking is learned from the inner training subset only.
#'   3. The top `k` features are retained.
#'   4. Median imputation, centering, and scaling parameters are learned from
#'      the inner training subset only.
#'   5. The same transformation is applied to the corresponding validation
#'      subset.
#'   6. The model is fitted using the candidate hyperparameters.
#'   7. Validation ROC AUC is calculated.
#'   8. ROC AUC values are averaged across inner folds.
#'
#' This design ensures that feature selection and preprocessing occur inside
#' cross-validation rather than before cross-validation, reducing information
#' leakage.
#'
#' The configuration with the highest mean inner ROC AUC is selected. If
#' multiple configurations have equivalent performance, configurations with
#' smaller `k` are preferred by the ordering rule.
inner_tune <- function(
  x,
  y,
  model_name,
  k_values,
  inner_splits = 4L,
  seed = 42L
) {

  folds <- make_folds(
    y,
    inner_splits,
    seed
  )

  grid <- expand_search_grid(
    model_name,
    k_values
  )

  grid$mean_inner_roc_auc <- NA_real_

  for (g in seq_len(nrow(grid))) {

    scores <- numeric(
      length(folds)
    )

    for (j in seq_along(folds)) {

      val <- folds[[j]]

      tr <- setdiff(
        seq_len(
          nrow(x)
        ),
        val
      )

      k <- min(
        as.integer(
          grid$k[g]
        ),
        ncol(x)
      )

      feats <- head(
        rank_features(
          x[
            tr,
            ,
            drop = FALSE
          ],
          y[tr]
        ),
        k
      )

      pp <- fit_preprocessor(
        x[
          tr,
          feats,
          drop = FALSE
        ]
      )

      xtr <- apply_preprocessor(
        x[
          tr,
          feats,
          drop = FALSE
        ],
        pp
      )

      xval <- apply_preprocessor(
        x[
          val,
          feats,
          drop = FALSE
        ],
        pp
      )

      params <- grid[
        g,
        setdiff(
          colnames(grid),
          c(
            "k",
            "mean_inner_roc_auc"
          )
        ),
        drop = FALSE
      ]

      fit <- fit_model(
        model_name,
        xtr,
        y[tr],
        params,
        seed + g + j
      )

      scores[j] <- safe_auc(
        y[val],
        predict_model_prob(
          model_name,
          fit,
          xval
        )
      )
    }

    grid$mean_inner_roc_auc[g] <- mean(
      scores,
      na.rm = TRUE
    )
  }

  grid[
    order(
      -grid$mean_inner_roc_auc,
      grid$k
    )[1],
    ,
    drop = FALSE
  ]
}


# ==============================================================================
# Nested cross-validation
# ==============================================================================


#' Evaluate one model using nested cross-validation
#'
#' Performs nested cross-validation in which the inner loop selects
#' hyperparameters and feature-set size and the outer loop estimates
#' generalization performance.
#'
#' @param x Predictor data frame.
#' @param y Binary outcome vector encoded as 0/1.
#' @param model_name Character machine-learning model name.
#' @param k_values Integer vector containing candidate feature-set sizes.
#' @param outer_splits Integer number of outer cross-validation folds.
#'   Defaults to 5.
#' @param inner_splits Integer number of inner cross-validation folds.
#'   Defaults to 4.
#' @param seed Integer random seed.
#'
#' @return A list containing:
#'
#'   \itemize{
#'     \item `fold_results`: Data frame containing performance metrics and
#'       tuning information for each outer fold.
#'     \item `selections`: List containing the features selected in each outer
#'       fold.
#'     \item `best_params`: List containing the best inner-CV configuration
#'       selected for each outer fold.
#'   }
#'
#' @details
#' For every outer fold:
#'
#'   1. The outer test fold is held out.
#'   2. `inner_tune()` is run using only the outer-training data.
#'   3. The selected number of features is ranked using only the outer-training
#'      observations.
#'   4. Preprocessing parameters are fitted using the outer-training data.
#'   5. The tuned model is trained using the complete outer-training partition.
#'   6. Predictions are generated for the untouched outer-test partition.
#'   7. Classification metrics are calculated from the outer-test predictions.
#'
#' Because the outer-test samples are not involved in feature selection,
#' preprocessing, or hyperparameter optimization, their results provide a less
#' biased estimate of model performance.
nested_cv_model <- function(
  x,
  y,
  model_name,
  k_values,
  outer_splits = 5L,
  inner_splits = 4L,
  seed = 42L
) {

  outer <- make_folds(
    y,
    outer_splits,
    seed
  )

  rows <- list()
  selections <- list()
  params_out <- list()

  for (i in seq_along(outer)) {

    te <- outer[[i]]

    tr <- setdiff(
      seq_len(
        nrow(x)
      ),
      te
    )

    best <- inner_tune(
      x[
        tr,
        ,
        drop = FALSE
      ],
      y[tr],
      model_name,
      k_values,
      inner_splits,
      seed + 1000L + i
    )

    k <- min(
      as.integer(
        best$k
      ),
      ncol(x)
    )

    feats <- head(
      rank_features(
        x[
          tr,
          ,
          drop = FALSE
        ],
        y[tr]
      ),
      k
    )

    pp <- fit_preprocessor(
      x[
        tr,
        feats,
        drop = FALSE
      ]
    )

    xtr <- apply_preprocessor(
      x[
        tr,
        feats,
        drop = FALSE
      ],
      pp
    )

    xte <- apply_preprocessor(
      x[
        te,
        feats,
        drop = FALSE
      ],
      pp
    )

    p <- best[
      ,
      setdiff(
        colnames(best),
        c(
          "k",
          "mean_inner_roc_auc"
        )
      ),
      drop = FALSE
    ]

    fit <- fit_model(
      model_name,
      xtr,
      y[tr],
      p,
      seed + i
    )

    prob <- predict_model_prob(
      model_name,
      fit,
      xte
    )

    rows[[i]] <- data.frame(
      model = model_name,
      outer_fold = i,
      inner_best_roc_auc = best$mean_inner_roc_auc,
      best_k = k,
      as.list(
        classification_metrics(
          y[te],
          prob
        )
      ),
      check.names = FALSE
    )

    selections[[i]] <- feats

    params_out[[i]] <- as.list(
      best
    )
  }

  list(
    fold_results = do.call(
      rbind,
      rows
    ),
    selections = selections,
    best_params = params_out
  )
}


# ==============================================================================
# Cross-validation result summarization
# ==============================================================================


#' Summarize outer-cross-validation performance
#'
#' Aggregates fold-level nested-cross-validation results for each model.
#'
#' @param d Data frame containing outer-fold results, typically produced by
#'   combining the `fold_results` objects returned by `nested_cv_model()`.
#'
#' @return Data frame containing one row per model with:
#'
#'   \itemize{
#'     \item Number of outer folds
#'     \item Mean of each classification metric
#'     \item Standard deviation of each metric
#'     \item Standard error of each metric
#'     \item Median selected feature count
#'   }
#'
#' @details
#' Metrics summarized are:
#'
#'   - ROC AUC
#'   - PR AUC / average precision
#'   - Accuracy
#'   - Balanced accuracy
#'   - Sensitivity
#'   - Specificity
#'   - Precision
#'   - F1 score
#'   - Brier score
summarize_outer_results <- function(d) {

  metrics <- c(
    "roc_auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "brier"
  )

  do.call(
    rbind,
    lapply(
      unique(
        d$model
      ),
      function(m) {

        x <- d[
          d$model == m,
          ,
          drop = FALSE
        ]

        row <- list(
          model = m,
          n_outer_folds = nrow(x)
        )

        for (metric in metrics) {

          vals <- as.numeric(
            x[[metric]]
          )

          row[[
            paste0(
              "mean_",
              metric
            )
          ]] <- mean(
            vals,
            na.rm = TRUE
          )

          row[[
            paste0(
              "sd_",
              metric
            )
          ]] <- sd(
            vals,
            na.rm = TRUE
          )

          row[[
            paste0(
              "se_",
              metric
            )
          ]] <- sd(
            vals,
            na.rm = TRUE
          ) /
            sqrt(
              sum(
                is.finite(vals)
              )
            )
        }

        row$median_best_k <- as.integer(
          median(
            x$best_k
          )
        )

        as.data.frame(
          row,
          check.names = FALSE
        )
      }
    )
  )
}


# ==============================================================================
# Model recommendation
# ==============================================================================


#' Recommend a final machine-learning algorithm
#'
#' Selects the preferred algorithm using mean outer-cross-validation ROC AUC
#' together with a one-standard-error-style eligibility rule.
#'
#' @param summary_table Model-summary data frame returned by
#'   `summarize_outer_results()`.
#'
#' @return Character name of the recommended machine-learning model.
#'
#' @details
#' The procedure is:
#'
#'   1. Identify the model with the highest mean outer-CV ROC AUC.
#'   2. Calculate a cutoff equal to:
#'
#'      best mean ROC AUC - standard error of the best model.
#'
#'   3. Treat all models with mean ROC AUC greater than or equal to this cutoff
#'      as eligible.
#'   4. Select the first eligible model according to `MODEL_ORDER`.
#'
#' Current preference order:
#'
#'   ElasticNet -> NaiveBayes -> SVM -> RandomForest
#'
#' The deterministic preference rule allows a simpler or preferred model to be
#' chosen when several algorithms have similar cross-validation discrimination.
recommend_model <- function(
  summary_table
) {

  best <- which.max(
    summary_table$mean_roc_auc
  )

  cutoff <-
    summary_table$mean_roc_auc[best] -
    summary_table$se_roc_auc[best]

  eligible <- summary_table$model[
    summary_table$mean_roc_auc >= cutoff
  ]

  for (m in MODEL_ORDER) {

    if (m %in% eligible) {
      return(m)
    }
  }

  summary_table$model[best]
}


# ==============================================================================
# Final frozen-model fitting
# ==============================================================================


#' Fit the final frozen classification model
#'
#' Tunes and trains the selected machine-learning algorithm using the complete
#' development dataset.
#'
#' @param x Predictor data frame containing the complete development dataset.
#' @param y Binary outcome vector encoded as 0/1.
#' @param model_name Character name of the selected machine-learning algorithm.
#' @param k_values Integer vector containing candidate feature-set sizes.
#' @param inner_splits Integer number of cross-validation folds used to select
#'   final hyperparameters and feature count. Defaults to 5.
#' @param seed Integer random seed.
#'
#' @return A list containing:
#'
#'   \itemize{
#'     \item `model`: Final fitted model object.
#'     \item `model_name`: Selected algorithm name.
#'     \item `selected_features`: Final selected predictor names.
#'     \item `preprocessor`: Fitted imputation, centering, and scaling
#'       parameters.
#'     \item `best_search_params`: Selected feature count and model
#'       hyperparameters.
#'     \item `inner_cv_roc_auc`: Mean inner-CV ROC AUC associated with the
#'       selected tuning configuration.
#'   }
#'
#' @details
#' This function is intended to be called after model-family evaluation has been
#' completed using nested cross-validation.
#'
#' The complete development dataset is used to:
#'
#'   1. Tune the selected model family using inner cross-validation.
#'   2. Rank predictors.
#'   3. Select the final top-k feature set.
#'   4. Fit preprocessing parameters.
#'   5. Transform the complete development dataset.
#'   6. Fit the final model.
#'
#' The returned `selected_features` and `preprocessor` must be preserved and
#' reused when applying the frozen model to independent validation or external
#' datasets.
fit_frozen_model <- function(
  x,
  y,
  model_name,
  k_values,
  inner_splits = 5L,
  seed = 42L
) {

  best <- inner_tune(
    x,
    y,
    model_name,
    k_values,
    inner_splits,
    seed
  )

  k <- min(
    as.integer(
      best$k
    ),
    ncol(x)
  )

  feats <- head(
    rank_features(
      x,
      y
    ),
    k
  )

  pp <- fit_preprocessor(
    x[
      ,
      feats,
      drop = FALSE
    ]
  )

  xp <- apply_preprocessor(
    x[
      ,
      feats,
      drop = FALSE
    ],
    pp
  )

  p <- best[
    ,
    setdiff(
      colnames(best),
      c(
        "k",
        "mean_inner_roc_auc"
      )
    ),
    drop = FALSE
  ]

  fit <- fit_model(
    model_name,
    xp,
    y,
    p,
    seed
  )

  list(
    model = fit,
    model_name = model_name,
    selected_features = feats,
    preprocessor = pp,
    best_search_params = as.list(
      best
    ),
    inner_cv_roc_auc = best$mean_inner_roc_auc
  )
}


# ==============================================================================
# Output utilities
# ==============================================================================


#' Write an object to a formatted JSON file
#'
#' Serializes an R object to JSON for storing model metadata, configuration,
#' performance summaries, or other workflow results.
#'
#' @param path Character output file path.
#' @param payload R object to serialize.
#'
#' @return The value returned by `jsonlite::write_json()`, invisibly when
#'   applicable.
#'
#' @details
#' JSON output is formatted for readability. Scalar values are automatically
#' unboxed, and R NULL values are represented as JSON `null`.
write_json <- function(
  path,
  payload
) {

  jsonlite::write_json(
    payload,
    path,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null"
  )
}