if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, 
               caret, DT, data.table, lightgbm, xgboost, magrittr, 
               tictoc,MLmetrics, caTools,recipes,rsample,
               EnvStats,cattonum,DataExplorer,corrplot,lubridate,catboost)

#########################################
VAR_SUM <- function(X,y,tri) {
  w <- cor.test(X[tri],y)
  print(w)
  q <- chisq.test(X[tri],y)
  print(q)
  z <- colAUC(X[tri],y)
  cat("AUC:",z,"\n")
}
#########################################
binary_plot <- function(x) { 
  
  ggplot(tr, aes(x = x, fill = target)) +
    geom_density(alpha=0.5, aes(fill=factor(target))) + labs(title="Density Plot") +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + theme_grey()
  
}
#########################################
AUC <- function(actual,predicted){
  r <- as.numeric(rank(predicted))
  n_pos <- as.numeric(sum(actual == 1))
  n_neg <- as.numeric(length(actual) - n_pos)
  auc <- (sum(r[actual == 1]) - n_pos * (n_pos + 1)/2)/(n_pos *  n_neg)
  auc
}
#########################################
#Preprocessed, frequency encoded dataset
#tr_te <- read_csv("tr_te_top200.csv")

###########################################
#Load data, OOF optional

COMBINE_DATA <- function(oof1 = FALSE, oof2 = FALSE){
  tr <- read_csv("train.csv") 
 
  te <- read_csv("test.csv") 
  ###########OOF##########################
  if(oof1 == TRUE){
    
    ###########Train
    #lgbm
    x <- read_csv("oof_tr_lgbtree_cv-0.9007366_2019-3-9.csv")
    tr$pred1 <- x$pred_tr_rf

    x <- read_csv("oof_tr_lgbtree_cv-0.9004574_2019-3-8.csv") ##better of the two
    tr$pred2 <- x$pred_tr_rf
    
    x <- read_csv("oof_tr_lgbtree_cv-0.9013579_2019-3-20.csv")
    tr$pred3 <- x$pred_tr_rf
    
    x <- read_csv("cv_0.90166_lgb_oof_2019-3-20.csv")
    tr$pred4 <- x$predict
    
    x <-read_csv("oof_tr_lgbtree_cv-0.9039104_2019-4-06.csv")
    tr$pred5 <- x$pred_tr_rf 
    
    x <- read_csv("oof_tr_lgbtree_cv-0.9067953_2019-4-06.csv")
    tr$pred6 <- x$pred_tr_rf
    
    tr$pred7 <- pred_tr_rf
    
    #xgb
    z <- read_csv("oof_tr_xgb_cv-0.898174.csv")
    tr$pred_xgb <- z$pred_tr_xgb
    
    ###########Test
    #lgbm
    q <- read_csv("oof_te_lgbtree_cv-0.9007366_2019-3-9.csv")
    te$pred1 <- q$pred_te_rf
    # 
    q <- read_csv("oof_te_lgbtree_cv-0.9004574_2019-3-8.csv") 
    te$pred2 <- q$pred_te_rf
    
    q <- read_csv("oof_te_lgbtree_cv-0.9013579_2019-3-20.csv")
    te$pred3 <- q$pred_te_rf
    
    q <- read_csv("cv_0.90166_lgb_submission.csv")
    te$pred4 <- q$target
    
    q <-read_csv("oof_te_lgbtree_cv-0.9039104_2019-4-06.csv")
    te$pred5 <- q$pred_te_rf
    
    q <- read_csv("oof_te_lgbtree_cv-0.9067953_2019-4-06.csv")
    te$pred6 <- q$pred_te_rf
    
    te$pred7 <- pred_te_rf
    
    ##xgb
    w <- read_csv("oof_te_xgb_cv-0.898174.csv")
    te$pred_xgb <- w$pred_te_xgb
    
  }
  
  y <- tr$target
  tri <- 1:nrow(tr)
  
  tr_te <- tr %>%
    dplyr::select(-target) %>%
    bind_rows(te) %>%
    dplyr::select(-ID_code) 
  

  if(oof2 == TRUE){
    #catboost
    s <- read_csv("pred_tr_cat_cv-0.8988213.csv")
    tr_te$catboost <- s$x
    
    t <- read_csv("kerasRstack4-6.csv")
    tr_te$keras <- t$x
    
    lda <- read_csv("lda_preds.csv")
    tr_te$lda <- lda$posterior.1
    
  }
  

  tr_te <<- tr_te
  y <<- y
  tri <<- tri
}
############################################

