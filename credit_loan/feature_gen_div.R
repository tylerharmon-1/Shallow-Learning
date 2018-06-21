##################################################################
###############-----AUTO_FEATURE_GENERATOR----------##############
##WE WANT TO SEE A SIMPLE VIEW INTO THE IMPORTANCE PLOT OF THE NEW VARIABLES
##################################################################
library(tidyverse)
library(magrittr)
library(caret)
library(lightgbm)
library(knitr)
##########################
#######loop that creates new features using "/" operator#####

##DATA FILE##
source("~/projects/kaggle/home_credit_default_risk/feat_gen_data.R")
#############################
###MOVE 1ST COLUMN TO LAST###

new_fg = fg %>%
  select(-NAME_CONTRACT_TYPE, everything())

##############################################
col_length = ncol(new_fg)
div_col_length = as.numeric(col_length) - 1 
row_length = nrow(new_fg)

new_df = matrix(ncol = (div_col_length), nrow = row_length)
###########
##########Performs 1 epoch of the dataset
for (i in 1:row_length) {
  for (j in 1:div_col_length) {
    new_df[i,j] <- new_fg[i,j] / new_fg[i,j+1]
    
  }
  
} 
###set column names
colnames(new_df) <- paste("col2by", 3:597, sep = "")
#####################################################
# feat_gen_div <- function(df,i,j) df[i,j] / df[i,j+1]
# nf <- apply(tr_te,1,feat_gen_div,i=1:356255,j=1:599)
out <- new_df %>%
  as.data.frame() %>%
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>%
  mutate_all(funs(ifelse(is.infinite(.), NA, .))) 

med <- apply(out,2,median,na.rm=TRUE)
out <-Impute(out, med)

##change name of last col to 1
out$col2by1 = out$col2by597
out$col2by597 = NULL
#######################################################
dtest <- out[-tri, ]
tr_te <- out[tri, ]
set.seed(123)
tri <- caret::createDataPartition(y, p = 0.9, list = F) %>% c() #0.84


lgb.train = lgb.Dataset(data.matrix(tr_te[tri,]), label = y[tri])
lgb.valid = lgb.Dataset(data.matrix(tr_te[-tri,]), label = y[-tri])

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
#set.seed(454545)
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
#############################################################################

#Importance of the Variables via kable & plot
# get feature importance
lgb.importance(lgb.model, percentage = TRUE) %>% head(20) %>% kable()
tree_imp <- lgb.importance(lgb.model, percentage = TRUE) %>% head(30)
lgb.plot.importance(tree_imp, top_n = 30, measure = "Gain")
###########################################################################
###Possibles: 2018-06-20
            # gender_to_apartments_avg = CODE_GENDER / APARTMENTS_AVG
            # gender_to_annuity_ratio = CODE_GENDER / AMT_ANNUITY
            # gender_to_employed = CODE_GENDER / DAYS_EMPLOYED
