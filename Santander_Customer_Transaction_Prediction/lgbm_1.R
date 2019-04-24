
##############
cat("CV and Model Training...")
start_time <- Sys.time()

#Importance Plot
plot_imp <- function(imp, title) {
  imp %>% 
    group_by(Feature) %>% 
    summarise_all(funs(mean)) %>% 
    arrange(desc(Gain)) %>% 
    top_n(50, Gain) %>% 
    ungroup() %>% 
    ggplot(aes(reorder(Feature, Gain), Gain)) + 
    geom_col(fill = "steelblue") +
    xlab("Feature") +
    ggtitle(title) +
    coord_flip() +
    theme_minimal()
}
###################################################################################################
###################################################################################################
nfolds <- 5
cv_score <- rep(0,nfolds)

set.seed(2000)

skf <- caret::createFolds(y, k = nfolds)
pred_tr_rf <- rep(0, nrow(tr_te[tri, ]))
pred_te_rf <- rep(0, nrow(tr_te[-tri, ]))
imp <- tibble(Feature = colnames(tr_te), Gain = 0, Cover = 0, Frequency = 0)

p <- list(objective = "binary",
          boost="gbdt",
          metric="auc",
          boost_from_average="false",
          num_threads=4,
          learning_rate = 0.01,
          num_leaves = 5, #5 (0.9111739) #6 (0.9108531) ;#13,
          max_depth=-1,
          tree_learner = "serial",
          feature_fraction = 0.9, #0.9098498, #0.3 (0.9081319) #0.2 (0.9076896), #0.15 (0.9066746), #0.05 (0.905194),
          bagging_freq = 5,#6 (0.9101333), #5,
          bagging_fraction = 0.4,
          min_data_in_leaf = 80,
          min_sum_hessian_in_leaf = 10.0,
          verbosity = 1)

for (i in seq_along(skf)){
  cat("\nFold:", i, "\n")
  idx <- skf[[i]]
  
  xtrain <- lgb.Dataset(data = data.matrix(tr_te[tri, ][-idx, ]), label = y[-idx], free_raw_data = FALSE)
  xval <- lgb.Dataset(data = data.matrix(tr_te[tri, ][idx, ]), label = y[idx], free_raw_data = FALSE)
  
  m_rf <- lgb.train(p, 
                    xtrain, 
                    nrounds = 1000000, 
                    list(val = xval), 
                    eval_freq = 500,
                    early_stopping_rounds = 2000, 
                    seed = 44000)
  
  
  cv_score[i] <- m_rf$best_score
  cat("Best Score = ", m_rf$best_score, '\n')
  
  pred_tr_rf[idx] <- predict(m_rf, data.matrix(tr_te[tri, ][idx, ]))
  pred_te_rf <- pred_te_rf + predict(m_rf, data.matrix(tr_te[-tri, ])) / nfolds
  imp %<>% bind_rows(lgb.importance(m_rf))
  
  rm(m_rf, xtrain, xval); invisible(gc())
}

cat("Average CV Score = ", mean(cv_score), '\n')
plot_imp(imp, "GBTree")

# read_csv("sample_submission.csv") %>%
#   mutate(target = pred_te_rf) %>%
#   write_csv("lgbtree_cv-0.9027883_2019-4-09.csv")

# write_csv(as.data.frame(pred_tr_rf),"oof_tr_lgbtree_cv-0.9083051_2019-4-10.csv")
# write_csv(as.data.frame(pred_te_rf),"oof_te_lgbtree_cv-0.9083051_2019-4-10.csv")
end_time <- Sys.time()

time_diff <- end_time - start_time
cat("Modeling Time Elapsed (Time wasted!): ", time_diff, "\n")
#######################################################################
