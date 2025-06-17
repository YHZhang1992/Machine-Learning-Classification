# CRAN & Bioconductor
install.packages(c("caret", "randomForest", "e1071", "naivebayes", "glmnet"))
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("limma")  # Optional for feature filtering

# For deep learning
install.packages("keras")
library(keras)
install_keras()  # Only once, with TensorFlow backend

# Load libraries
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(naivebayes)
library(keras)

# Example binary classification data
# Assume data has features in columns and binary class in 'label'
set.seed(42)
n <- 200
p <- 100
data <- data.frame(matrix(rnorm(n * p), nrow = n, ncol = p))
colnames(data) <- paste0("Gene", 1:p)
data$label <- factor(sample(c("Case", "Control"), n, replace = TRUE))

# Split into features and label
X <- data[, -ncol(data)]
y <- data$label

# feature selection method (replaceable)
feature_selection <- function(data, outcome, method = "t-test", top_n = 30) {
  if (method == "t-test") {
    pvals <- apply(data, 2, function(x) t.test(x ~ outcome)$p.value)
    selected <- names(sort(pvals))[1:top_n]
  } else {
    selected <- colnames(data)  # fallback
  }
  return(selected)
}

folds <- createFolds(y, k = 10, list = TRUE, returnTrain = FALSE)

results <- list()

for (i in 1:10) {
  test_idx <- folds[[i]]
  train_data <- X[-test_idx, ]
  test_data <- X[test_idx, ]
  train_label <- y[-test_idx]
  test_label <- y[test_idx]

  # ----- Feature Selection -----
  selected_features <- feature_selection(train_data, train_label, top_n = 30)
  train_selected <- train_data[, selected_features]
  test_selected <- test_data[, selected_features]

  # ----- Elastic Net -----
  ctrl <- trainControl(method = "cv", number = 5)
  elastic_model <- train(train_selected, train_label,
                         method = "glmnet",
                         trControl = ctrl,
                         tuneLength = 10)

  pred_elastic <- predict(elastic_model, test_selected)

  # ----- Random Forest -----
  rf_model <- train(train_selected, train_label,
                    method = "rf",
                    trControl = ctrl,
                    tuneLength = 5)

  pred_rf <- predict(rf_model, test_selected)

  # ----- Naive Bayes -----
  nb_model <- train(train_selected, train_label,
                    method = "naive_bayes",
                    trControl = ctrl)

  pred_nb <- predict(nb_model, test_selected)

  # ----- SVM -----
  svm_model <- train(train_selected, train_label,
                     method = "svmRadial",
                     trControl = ctrl,
                     tuneLength = 5)

  pred_svm <- predict(svm_model, test_selected)

  # ----- CNN -----
  x_train_cnn <- as.matrix(train_selected)
  x_test_cnn <- as.matrix(test_selected)
  x_train_cnn <- array_reshape(x_train_cnn, c(nrow(x_train_cnn), ncol(x_train_cnn), 1))
  x_test_cnn <- array_reshape(x_test_cnn, c(nrow(x_test_cnn), ncol(x_test_cnn), 1))

  y_train_cnn <- as.numeric(train_label == "Case")
  y_test_cnn <- as.numeric(test_label == "Case")

  cnn_model <- keras_model_sequential() %>%
    layer_conv_1d(filters = 16, kernel_size = 3, activation = 'relu', input_shape = c(dim(x_train_cnn)[2], 1)) %>%
    layer_global_max_pooling_1d() %>%
    layer_dense(units = 10, activation = 'relu') %>%
    layer_dense(units = 1, activation = 'sigmoid')

  cnn_model %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_adam(),
    metrics = "accuracy"
  )

  cnn_model %>% fit(x_train_cnn, y_train_cnn,
                    epochs = 20, batch_size = 16, verbose = 0)

  pred_cnn <- predict(cnn_model, x_test_cnn)
  pred_cnn <- ifelse(pred_cnn > 0.5, "Case", "Control")
  pred_cnn <- factor(pred_cnn, levels = c("Case", "Control"))

  # ----- Store Predictions -----
  results[[i]] <- list(
    true = test_label,
    elastic = pred_elastic,
    rf = pred_rf,
    nb = pred_nb,
    svm = pred_svm,
    cnn = pred_cnn,
    features = selected_features
  )
}

library(caret)
all_preds <- lapply(results, function(r) {
  data.frame(
    True = r$true,
    Elastic = r$elastic,
    RF = r$rf,
    NB = r$nb,
    SVM = r$svm,
    CNN = r$cnn
  )
})

combined <- do.call(rbind, all_preds)

# Accuracy by model
sapply(combined[,-1], function(pred) mean(pred == combined$True))

feature_usage <- table(unlist(lapply(results, `[[`, "features")))
top_used <- sort(feature_usage, decreasing = TRUE)
print(top_used[1:10])
