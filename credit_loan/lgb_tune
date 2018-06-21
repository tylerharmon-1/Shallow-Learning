
###########################################################################################
###########################################################################################
#######################################NOTES###############################################
###########################################################################################
#####DATA FILES#####
#source("~/projects/kaggle/home_credit_default_risk/lgb_data.R")
#source("~/projects/kaggle/home_credit_default_risk/lgb_data_dummies.R")
source("~/projects/kaggle/home_credit_default_risk/lgb_data_2.0.R")

library(lightgbm)
library(caret)
library(tidyverse)
##############################Data Partition################################################## 
####2018-6-20: Consider changing partition to 0.84 again, might be better for overall test set.

dtest <- tr_te[-tri, ]
tr_te <- tr_te[tri, ]
set.seed(123)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c() #0.84


lgb.train = lgb.Dataset(data.matrix(tr_te[tri,]), label = y[tri])
lgb.valid = lgb.Dataset(data.matrix(tr_te[-tri,]), label = y[-tri])

##################################################################################
#################################grid search######################################
# grid_search <- expand.grid(#max_depth = c(10,20,40,80),
#                            min_data_in_leaf = c(1,2),
#                            min_sum_hessian_in_leaf = c(50,100),
#                            feature_fraction = c(0.8,0.9,0.95,1.0),
#                            bagging_fraction = c(0.6,0.8,1.0),
#                            bagging_freq = c(0,1),
#                            lambda_l1 = c(0.2,0.4),
#                            num_leaves = c(7,8,9,10,20,30)
#                            #lamda_l2 = c(0.2,0.4),
#                            #min_gain_to_split = c(0.2,0.4)
#                            )
# 
# perf <- numeric(nrow(grid_search))
# 
# library(data.table)
# for (i in 1:nrow(grid_search)) {
#   fit_cv <- lgb.cv(list(objective = "binary",
#                         metric = "auc",
#                         #max_depth = grid_search[i, "max_depth"],
#                         min_data_in_leaf = grid_search[i, "min_data_in_leaf"],
#                         min_sum_hessian_in_leaf = grid_search[i, "min_sum_hessian_in_leaf"],
#                         feature_fraction = grid_search[i, "feature_fraction"],
#                         bagging_fraction = grid_search[i, "bagging_fraction"],
#                         bagging_freq = grid_search[i, "bagging_freq"],
#                         lambda_l1 = grid_search[i, "lambda_l1"],
#                         num_leaves = grid_search[i, "num_leaves"]
#                         #lambda_l2 = grid_search[i, "lambda_l2"],
#                         #min_gain_to_split = grid_search[i, "min_gain_to_split"]
#                         ),
#                    data = lgb.train,
#                    nrounds = 3000,
#                    nfold = 5,
#                    early_stopping_rounds = 200,
#                    verbose = 1
#                    #num_leaves = 7
#                    )
# 
#   perf[i] <- max(rbindlist(fit_cv$record_evals$valid$auc))
#   gc(verbose = FALSE)
# }
# cat("Model ", which.max(perf), " is highest auc: ", max(perf), sep = "" ,"\n")
# print(grid_search[which.max(perf), ])
# 
# fit_cv$best_iter ##best iteration 754
# fit_cv$best_score
##################################################################################
params.lgb = list(
  objective = "binary"
  , metric = "auc"
  , min_data_in_leaf = 2  #1   
  , min_sum_hessian_in_leaf = 100 #50 
  , feature_fraction = 0.95  #1 #0.9 
  , bagging_fraction = 1 #0.8 #1
  , bagging_freq = 1 #0
  , lambda_l1 = 0.2
)
#############################################################################
####Bayesian optimization: 2018-06-17, changed learning rate to 0.02, num_leaves to 34; need to check if is case for
###overfitting---cv score = 0.79288; LB = 0.790
set.seed(123)
  lgb.model <- lgb.train(
    params = params.lgb
    , data = lgb.train
    , valids = list(validation = lgb.valid)#,train = lgb.train)
    , learning_rate = 0.02 #0.05
    , num_leaves = 34 #7
    , num_threads = 2
    , nrounds = 4000#3000 #754 #1714
    , early_stopping_rounds = 200
    , eval_freq = 50
  )

 
lgb.model$best_score
lgb.model$best_iter
##########################################################################
set.seed(454545)
lgb.model_2 <- lgb.train(
  params = params.lgb
  , data = lgb.train
  , valids = list(validation = lgb.valid)#,train = lgb.train)
  , learning_rate = 0.02 #0.05
  , num_leaves = 34 #7
  , num_threads = 2
  , nrounds = 1303 #1714#2276 #1640 #2000 #754
  , early_stopping_rounds = 200
  , eval_freq = 50
)
#############################################################################
set.seed(2018)
lgb.model_3 <- lgb.train(
  params = params.lgb
  , data = lgb.train
  , valids = list(validation = lgb.valid)#,train = lgb.train)
  , learning_rate = 0.02 #0.05
  , num_leaves = 34 #7
  , num_threads = 2
  , nrounds = 1303 #1714#1640 #2000 #754
  , early_stopping_rounds = 200
  , eval_freq = 50
)
#############################################################################
#Importance of the Variables via kable & plot
# get feature importance
lgb.importance(lgb.model, percentage = TRUE) %>% head(20) %>% kable()
tree_imp <- lgb.importance(lgb.model, percentage = TRUE) %>% head(40)
lgb.plot.importance(tree_imp, top_n = 30, measure = "Gain")
###########################################################################
###########################ENSEMBLE########################################
#Make the prediction and prepare for the submission.
# make test predictions
#test=read.csv("~/projects/kaggle/home_credit_default_risk/application_test.csv")  #%>% data.matrix()
lgb_pred_1 <- predict(lgb.model, data = data.matrix(dtest), n = lgb.model$best_iter)
lgb_pred_2 <- predict(lgb.model_2, data = data.matrix(dtest), n = lgb.model_2$best_iter)
lgb_pred_3 <- predict(lgb.model_3, data = data.matrix(dtest), n = lgb.model_3$best_iter)

lgb_pred <- (lgb_pred_1+lgb_pred_2+lgb_pred_3)/3
#############################################################################  
  read_csv("~/projects/kaggle/home_credit_default_risk/sample_submission.csv") %>%  
    mutate(SK_ID_CURR = as.integer(SK_ID_CURR),
           TARGET = lgb_pred) %>%
    write_csv(paste0("~/projects/kaggle/home_credit_default_risk/lgb_2018-6-20_", round(lgb.model$best_score,4), ".csv"))
  
###################################################################################################
  
