required <- c("caret", "glmnet", "randomForest", "e1071", "pROC", "jsonlite")
missing <- required[!vapply(required, requireNamespace, quietly = TRUE, FUN.VALUE = logical(1))]
if (length(missing)) install.packages(missing)
