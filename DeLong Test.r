# -*- coding: utf-8 -*-

# -- Sheet --

if (!requireNamespace("pROC", quietly = TRUE)) {
    install.packages("pROC")
}
library(pROC)

data_for_r <- read.csv('data_for_r.csv')

library(pROC)

roc_obj1 <- roc(data_for_r$y_test, data_for_r$y_pred1)
roc_obj2 <- roc(data_for_r$y_test, data_for_r$y_pred2)

delong_results <- roc.test(roc_obj1, roc_obj2, method="delong")

print(delong_results)

data_for_r <- read.csv('data_for_r2.csv')

library(pROC)

roc_obj1 <- roc(data_for_r$y_test, data_for_r$y_pred1)
roc_obj2 <- roc(data_for_r$y_test, data_for_r$y_pred3)

delong_results <- roc.test(roc_obj1, roc_obj2, method="delong")

print(delong_results)

data_for_r <- read.csv('data_for_r3.csv')

library(pROC)

roc_obj1 <- roc(data_for_r$y_test, data_for_r$y_pred2)
roc_obj2 <- roc(data_for_r$y_test, data_for_r$y_pred3)

delong_results <- roc.test(roc_obj1, roc_obj2, method="delong")

print(delong_results)



