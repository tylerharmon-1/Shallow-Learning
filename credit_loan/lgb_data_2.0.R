library(tidyverse)
library(magrittr)
library(caret)
library(lightgbm)
library(knitr)

#####################################################################
#####################################################################
#### This function imputes each feature with some constant values
Impute <- function(data, value){
  
  num <- apply(data, 2, function(x) sum(is.na(x)))
  
  data <- as.matrix(data)
  data[which(is.na(data))] <- rep(value, num)
  data <- as.data.frame(data)
  
  return(data)
}
####################################################################
#######################################################################
###Read in data

tr <- read_csv("~/projects/kaggle/home_credit_default_risk/application_train.csv") 
te <- read.csv("~/projects/kaggle/home_credit_default_risk/application_test.csv")
bureau <- read_csv("~/projects/kaggle/home_credit_default_risk/bureau.csv") 
bbalance <- read_csv("~/projects/kaggle/home_credit_default_risk/bureau_balance.csv") 
cc_balance <-  read_csv("~/projects/kaggle/home_credit_default_risk/credit_card_balance.csv") 
pc_balance <- read_csv("~/projects/kaggle/home_credit_default_risk/POS_CASH_balance.csv")
prev <- read_csv("~/projects/kaggle/home_credit_default_risk/previous_application.csv") 
payments <- read_csv("~/projects/kaggle/home_credit_default_risk/installments_payments.csv") 

#######################################################################
###Preprocess data

fn <- funs(mean, sd, min, max, n_distinct, .args = list( na.rm =  TRUE))

sum_bbalance <- bbalance %>%
  mutate_if(is.character , funs(factor(.) %>% as.integer)) %>%
  group_by(SK_ID_BUREAU) %>%
  summarise_all(fn)

sum_bureau <- bureau %>% 
  left_join(sum_bbalance, by = "SK_ID_BUREAU") %>% 
  select(-SK_ID_BUREAU) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)

sum_cc_balance <- cc_balance %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)

####################################################################
########NO IMPROVEMENT
#payment_diff = AMT_INSTALMENT - AMT_PAYMENT) 
#dbd = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
#dbd = ifelse(dbd < 0, 0, dbd))
####################################################################

sum_payments <- payments %>%
  select(-SK_ID_PREV) %>%
  mutate(payment_perc = AMT_PAYMENT / AMT_INSTALMENT) %>% #2018-06-19--improved cv:0.793173
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)

sum_pc_balance <- pc_balance %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)


# Add feature: value ask / value received percentage
#prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
sum_prev <- prev %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
         DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
         DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
         DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
         DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
         APP_CREDIT_PERC = AMT_APPLICATION / AMT_CREDIT) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)
  
#########################################
#####################################################
tri <- 1:nrow(tr)
y <- tr$TARGET
###################################################
tr_te <- tr %>%
  select(-TARGET) %>%
  bind_rows(te) %>%
  left_join(sum_bureau, by = "SK_ID_CURR") %>%
  left_join(sum_cc_balance, by = "SK_ID_CURR") %>%
  left_join(sum_payments, by = "SK_ID_CURR") %>%
  left_join(sum_pc_balance, by = "SK_ID_CURR") %>%
  left_join(sum_prev, by = "SK_ID_CURR") %>%
  select(-SK_ID_CURR) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(na = apply(., 1, function(x) sum(is.na(x))),
         DAYS_EMPLOYED = ifelse(DAYS_EMPLOYED == 365243, NA, DAYS_EMPLOYED),
         DAYS_EMPLOYED_PERC = DAYS_EMPLOYED / DAYS_BIRTH,
         INCOME_CREDIT_PERC = AMT_INCOME_TOTAL / AMT_CREDIT,
         INCOME_PER_PERSON = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS,
         ANNUITY_INCOME_PERC = AMT_ANNUITY / AMT_INCOME_TOTAL,
         LOAN_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
         ANNUITY_INCOME_RATIO = AMT_ANNUITY / AMT_INCOME_TOTAL, 
         ANNUITY_LENGTH = AMT_CREDIT / AMT_ANNUITY,
         WORKING_LIFE_RATIO = DAYS_EMPLOYED / DAYS_BIRTH,
         CHILDREN_RATIO = CNT_CHILDREN / CNT_FAM_MEMBERS,
         name_contract_by_apartments_avg = NAME_CONTRACT_TYPE / APARTMENTS_AVG, #cv:0.7929809 ##1by43
         CREDIT_TO_GOODS_RATIO = AMT_CREDIT / AMT_GOODS_PRICE, #2018-06-18--cv:0.7930399; LB:0.794
         phone_to_employ_ratio = DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED ##2018-06-21--cv:0.7934275
         
         ) %>%  
  
  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>%
  mutate_all(funs(ifelse(is.infinite(.), NA, .)))   
####################################################################
#######Impute (median) NAs
med <- apply(tr_te,2,median,na.rm=TRUE)
tr_te <-Impute(tr_te, med)
####################################################################
#################FEATURES TO TRY#################################
# 








#################NO IMPROVEMENT FEATURES#########################
#CAR_TO_BIRTH_RATIO = OWN_CAR_AGE / DAYS_BIRTH ---no improvement
#name_contract_to_days_credit_m = NAME_CONTRACT_TYPE / DAYS_CREDIT_mean -- improvement
#name_contract_to_amt_goods_price = NAME_CONTRACT_TYPE / AMT_GOODS_PRICE
#DAYS_EMPLOYED_PERC_NRM = sqrt(DAYS_EMPLOYED / DAYS_BIRTH)
#INCOME_PER_PERSON_NRM = log1p(AMT_INCOME_TOTAL / CNT_FAM_MEMBERS)
#ANNUITY_INCOME_PERC_NRM = sqrt(AMT_ANNUITY / AMT_INCOME_TOTAL
#CAR_TO_EMPLOY_RATIO = OWN_CAR_AGE / DAYS_EMPLOYED
#gender_to_apartments_avg = CODE_GENDER / APARTMENTS_AVG
#gender_to_ext2 = CODE_GENDER / EXT_SOURCE_2 ##2by41
#gender_to_annuity_ratio = CODE_GENDER / AMT_ANNUITY ##2by8
#name_contract_to_ext2 = NAME_CONTRACT_TYPE / EXT_SOURCE_2
#phone_to_birth_ratio = DAYS_LAST_PHONE_CHANGE / DAYS_BIRTH
#gender_to_employed = CODE_GENDER / DAYS_EMPLOYED
######################################################################
