if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, skimr, GGally, plotly, viridis, 
               caret, DT, data.table, lightgbm, xgboost, magrittr, tictoc,MLmetrics, caTools,recipes,rsample,
               EnvStats,cattonum)

gc()
start_time <- Sys.time()
#########################################
summaryFunction <- function(X,y,tri) {
  w <- cor.test(X[tri],y)
  print(w)
  q <- chisq.test(X[tri],y)
  print(q)
  z <- colAUC(X[tri],y)
  print(z)
  # plot(y~X[tri])
}
#########################################
credit_plot <- function(x) { 
  
  ggplot(tr_te, aes(x = x, fill = TARGET)) +
    geom_density(alpha=0.5, aes(fill=factor(TARGET))) + labs(title="Density of TARGET, given 
                                                             CREDIT_DIV_ANNUITY") +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 10)) + theme_grey()
  
}
##########################################
#---------------------------
cat("Loading data...\n")
out <- fread("~/Desktop/credit_loan/days_employed_div_data.csv")
days_employed_by_10 = out$DAYS_EMPLOYED_by_10
rm(out); gc()


###avg debt / amt income AMT_CREDIT_SUM_DEBT_mean+AMT_BALANCE_mean+AMT_PAYMENT_mean / AMT_INCOME_TOTAL

tr <- fread("~/projects/kaggle/home_credit_default_risk/application_train.csv") %>% as.data.frame()
te <- fread("~/projects/kaggle/home_credit_default_risk/application_test.csv") %>% as.data.frame()
bureau <- fread("~/projects/kaggle/home_credit_default_risk/bureau.csv") %>% as.data.frame()
bbalance <- fread("~/projects/kaggle/home_credit_default_risk/bureau_balance.csv") %>% as.data.frame()
cc_balance <-  fread("~/projects/kaggle/home_credit_default_risk/credit_card_balance.csv") %>% as.data.frame()
pc_balance <- fread("~/projects/kaggle/home_credit_default_risk/POS_CASH_balance.csv") %>% as.data.frame()
prev <- fread("~/projects/kaggle/home_credit_default_risk/previous_application.csv") %>% as.data.frame() 
payments <- fread("~/projects/kaggle/home_credit_default_risk/installments_payments.csv") %>% as.data.frame()

#---------------------------
cat("Preprocessing...\n")
#### This function imputes each feature with some constant values
Impute <- function(data, value){

  num <- apply(data, 2, function(x) sum(is.na(x)))

  data <- as.matrix(data)
  data[which(is.na(data))] <- rep(value, num)
  data <- as.data.frame(data)

  return(data)
}
#####################################################
##Preprocessing
# tr$employment <- ifelse(tr$DAYS_EMPLOYED == 365243,NA,tr$DAYS_EMPLOYED)
# te$employment <- ifelse(te$DAYS_EMPLOYED == 365243,NA,te$DAYS_EMPLOYED)

#####################################################
#mean encoding 0.7936503 
tri <- 1:nrow(tr)
df = bind_rows(list(tr,te))
####################################
# library(woeBinning)
# x <- woe.binning(df,target.var = 'TARGET',pred.var = c('DAYS_BIRTH'))
# df <- woe.binning.deploy(df,x)
####################################
df <- catto_onehot(df,c(CODE_GENDER,FLAG_OWN_CAR,FLAG_OWN_REALTY,NAME_CONTRACT_TYPE)) #cv:0.7939545
df <- catto_mean(df,response = TARGET,verbose = T)
# 
# #df<- catto_loo(df,response = TARGET,verbose = T)
te <- df[-tri,]
tr <- df[tri,]
# te$TARGET <- NULL
# 
rm(df,x)
#####################################################
######################################################
cat("Processing bureau balance...\n")
fn <- funs(mean, sd, min, max, sum, .args = list(na.rm = TRUE)) # removing n_distinct increases cv to 0.7919705

#create dummies
# dmy <- dummyVars(" ~ .", data = bbalance)
# bbalance <- predict(dmy, newdata = bbalance) %>% as.data.frame()
#bbalance <- catto_onehot(bbalance,verbose=T)

# sum_bbalance <- bbalance %>%
#   mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
#   group_by(SK_ID_BUREAU) %>% 
#   summarise_all(funs(mean,sd, .args=list(na.rm = TRUE)))  #mean,sd cv=0.7936021
##############################################################################
active_bbalance <- bbalance %>%
  filter(STATUS != "C" & STATUS !="X") %>%
  mutate(no_dpd = ifelse(STATUS == 0,1,0),
         dpd_1 = ifelse(STATUS == 1,1,0),
         dpd_2 = ifelse(STATUS == 2,1,0),
         dpd_3 = ifelse(STATUS == 3,1,0),
         dpd_4 = ifelse(STATUS == 4,1,0),
         dpd_5 = ifelse(STATUS == 5,1,0)) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  group_by(SK_ID_BUREAU) %>%
  summarise_all(funs(mean,sd, .args=list(na.rm = TRUE)))

##############################################################################
closed_bbalance <- bbalance %>%
  filter(STATUS == "C") %>%
  mutate(STATUS = ifelse(STATUS == "C",1,0)) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  group_by(SK_ID_BUREAU) %>%
  summarise_all(funs(mean,sd, .args=list(na.rm = TRUE)))

rm(bbalance); gc()
###########################################################################
cat("Processing bureau...\n")
#create dummies


sum_bureau_active <- bureau %>%
  left_join(active_bbalance, by = "SK_ID_BUREAU") %>%
  select(-SK_ID_BUREAU) %>%
  # group_by(CREDIT_ACTIVE) %>%
  filter(CREDIT_ACTIVE == "Active") %>%
  mutate(CREDIT_ACTIVE = ifelse(CREDIT_ACTIVE == "Active",1,0)) %>%
  catto_dummy() %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(days_cred_by_annuity = DAYS_CREDIT / AMT_ANNUITY,
         debt_credit_ratio = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM##very nice feature
         
         
  )%>%
  group_by(SK_ID_CURR) %>% #CREDIT_CURRENCY
  summarise_all(fn) #funs(mean, .args=list(na.rm = TRUE))

sum_bureau_closed <- bureau %>%
  left_join(closed_bbalance, by = "SK_ID_BUREAU") %>%
  select(-SK_ID_BUREAU) %>%
  # group_by(CREDIT_ACTIVE) %>%
  filter(CREDIT_ACTIVE == "Closed") %>%
  mutate(CREDIT_ACTIVE = ifelse(CREDIT_ACTIVE == "Closed",1,0)) %>%
  catto_dummy() %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(days_cred_by_annuity = DAYS_CREDIT / AMT_ANNUITY,
         debt_credit_ratio = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM##very nice feature
         
  )%>%
  group_by(SK_ID_CURR) %>% #CREDIT_CURRENCY
  summarise_all(fn) #funs(mean, .args=list(na.rm = TRUE))
rm(bureau, sum_bbalance,active_bbalance,closed_bbalance,active_bbalance_3mo,active_bbalance_6mo,
   active_bbalance_12mo,closed_bbalance_3mo,closed_bbalance_6mo,closed_bbalance_12mo); gc()

# sum_bureau <- bureau %>%
#   left_join(sum_bbalance, by = "SK_ID_BUREAU") %>%
#   select(-SK_ID_BUREAU) %>%
#   catto_dummy() %>%
#   mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
#   mutate(days_cred_by_annuity = DAYS_CREDIT / AMT_ANNUITY,
#          debt_credit_ratio = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM##very nice feature
# 
#          #debt_credit_util_ratio = AMT_CREDIT_SUM_DEBT / AMT_CREDIT_SUM_LIMIT
#          #bureau_credit_util_ratio = AMT_CREDIT_SUM / AMT_CREDIT_SUM_LIMIT -no improvement
#          )%>%
#   group_by(SK_ID_CURR) %>% #CREDIT_CURRENCY
#   summarise_all(fn) #funs(mean, .args=list(na.rm = TRUE))
# rm(bureau, sum_bbalance,dmy); gc()
#############################################################################
cat("Processing credit card...\n")
#create dummies

# sum_cc_balance <- cc_balance %>%
#   select(-SK_ID_PREV) %>%
#   mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
#   # mutate(
#   #   #cc_debt = AMT_BALANCE - AMT_PAYMENT_CURRENT
#   #   # cc_debt = ifelse(cc_debt<0,0,cc_debt)
#   #   ) %>% ###new
#   group_by(SK_ID_CURR) %>%
#   summarise_all(fn)
#######################################################################
#######################################################################

cc_3month <- cc_balance %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  filter(MONTHS_BALANCE >= -3) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)

cc_6month <- cc_balance %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  filter(MONTHS_BALANCE >= -6) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)

# 
cc_12month <- cc_balance %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  filter(MONTHS_BALANCE >= -12) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)



rm(cc_balance); gc()

cat("Processing payments...\n")
# #create dummies

# sum_payments <- payments %>%
#   select(-SK_ID_PREV) %>%
#   mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
#          PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
#          DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
#          DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT
#          # DPD = ifelse(DPD > 0, DPD, 0),
#          # DBD = ifelse(DBD > 0, DBD, 0),
#          # DPD_5  = ifelse(DPD <= 5,1,0),
#          # DPD_30 = ifelse(DPD < 30 & DPD > 5,1,0),
#          # DPD_90 = ifelse(DPD >= 90,1,0)
#          ) %>%
#   group_by(SK_ID_CURR) %>%
#   summarise_all(fn)
# rm(payments); gc()

sum_payments_5 <- payments %>%
  select(-SK_ID_PREV) %>%
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
         DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
         DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
         DPD = ifelse(DPD > 0, DPD, 0),
         DBD = ifelse(DBD > 0, DBD, 0)
         # DPD_5  = ifelse(DPD <= 5,1,0),
         # DPD_30 = ifelse(DPD < 30 & DPD > 5,1,0)
  ) %>%
  filter(DBD <= 5) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)

# sum_payments_30 <- payments %>%
#   select(-SK_ID_PREV) %>%
#   mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
#          PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
#          DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
#          DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
#          DPD = ifelse(DPD > 0, DPD, 0),
#          DBD = ifelse(DBD > 0, DBD, 0)
#          # DPD_5  = ifelse(DPD <= 5,1,0),
#          # DPD_30 = ifelse(DPD < 30 & DPD > 5,1,0)
#   ) %>%
#   filter(DBD <= 30) %>%
#   group_by(SK_ID_CURR) %>%
#   summarise_all(fn)

sum_payments_90 <- payments %>%
  select(-SK_ID_PREV) %>%
  mutate(PAYMENT_PERC = AMT_PAYMENT / AMT_INSTALMENT,
         PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT,
         DPD = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT,
         DBD = DAYS_INSTALMENT - DAYS_ENTRY_PAYMENT,
         DPD = ifelse(DPD > 0, DPD, 0),
         DBD = ifelse(DBD > 0, DBD, 0)
         # DPD_5  = ifelse(DPD <= 5,1,0),
         # DPD_30 = ifelse(DPD < 30 & DPD > 5,1,0)
  ) %>%
  filter(DBD <= 90) %>%
  group_by(SK_ID_CURR) %>%
summarise_all(fn)
rm(payments); gc()



cat("Processing pc balance...\n")
#create dummies
# dmy <- dummyVars("~.", data = pc_balance)
# pc_balance <- predict(dmy, newdata = pc_balance) %>% as.data.frame()
####################################################################
# sum_pc_balance <- pc_balance %>% 
#   select(-SK_ID_PREV) %>% 
#   mutate_if(is.character, funs(factor(.) %>% as.integer)) %>% 
#   mutate(
#         #cnt_total_instalment = CNT_INSTALMENT + CNT_INSTALMENT_FUTURE,
#         #cnt_instalment_remaining = CNT_INSTALMENT_FUTURE / cnt_total_instalment
#     ) %>%
#   group_by(SK_ID_CURR) %>% 
#   summarise_all(fn)
###################################################################
pc_3month <- pc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  filter(MONTHS_BALANCE >= -3) %>%
mutate(
  
  #cnt_total_instalment = CNT_INSTALMENT + CNT_INSTALMENT_FUTURE,
  #cnt_instalment_remaining = CNT_INSTALMENT_FUTURE / cnt_total_instalment
) %>%
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)

pc_6month <- pc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  filter(MONTHS_BALANCE >= -6) %>%
mutate(
  
  #cnt_total_instalment = CNT_INSTALMENT + CNT_INSTALMENT_FUTURE,
  #cnt_instalment_remaining = CNT_INSTALMENT_FUTURE / cnt_total_instalment
) %>%
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)

pc_12month <- pc_balance %>% 
  select(-SK_ID_PREV) %>% 
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  filter(MONTHS_BALANCE >= -12) %>%
mutate(
  
  #cnt_total_instalment = CNT_INSTALMENT + CNT_INSTALMENT_FUTURE,
  #cnt_instalment_remaining = CNT_INSTALMENT_FUTURE / cnt_total_instalment
) %>%
  group_by(SK_ID_CURR) %>% 
  summarise_all(fn)
###################################################################

rm(pc_balance); gc()

cat("Processing previous payments...\n")
# #create dummies
#######################################################
sum_prev <- prev %>%
  select(-SK_ID_PREV) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
  mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
         DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
         DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
         DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
         DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
         prev_app_credit_ratio = AMT_APPLICATION / AMT_CREDIT,
         prev_credit_annuity_ratio = AMT_CREDIT / AMT_ANNUITY

         #prev_late = DAYS_FIRST_DUE - DAYS_LAST_DUE_1ST_VERSION
         #prev_downpymt_cred_ratio = (AMT_APPLICATION - AMT_DOWN_PAYMENT) / AMT_CREDIT
         #prev_LTV = AMT_APPLICATION / AMT_GOODS_PRICE
         #prev_annuity_goods_ratio = AMT_ANNUITY / AMT_GOODS_PRICE ##no good
         #prev_annuity_down_payment_ratio = AMT_ANNUITY / AMT_DOWN_PAYMENT #no good
         #prev_credit_goods_ratio = AMT_CREDIT / AMT_GOODS_PRICE
         ) %>%
  group_by(SK_ID_CURR) %>%
  summarise_all(fn)


# x <-prev %>% count(SK_ID_CURR)
# 
# sum_prev$account_length <- x$n

  
#######################################################
# sum_active_prev <- prev %>%
#   select(-SK_ID_PREV) %>%
#   filter(NAME_CONTRACT_STATUS == "Approved") %>%
#   mutate(NAME_CONTRACT_STATUS = ifelse(NAME_CONTRACT_STATUS == "Approved",1,0)
#          ) %>%
#   mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
#   mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
#          DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
#          DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
#          DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
#          DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
#          prev_app_credit_ratio = AMT_APPLICATION / AMT_CREDIT,
#          prev_credit_annuity_ratio = AMT_CREDIT / AMT_ANNUITY
#   ) %>%  #new
#   group_by(SK_ID_CURR) %>%
# summarise_all(fn)

# sum_closed_prev <- prev %>%
#   select(-SK_ID_PREV) %>%
#   filter(NAME_CONTRACT_STATUS == "Refused") %>%
#   mutate(NAME_CONTRACT_STATUS = ifelse(NAME_CONTRACT_STATUS == "Refused",1,0)) %>%
#   mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
#   mutate(DAYS_FIRST_DRAWING = ifelse(DAYS_FIRST_DRAWING == 365243, NA, DAYS_FIRST_DRAWING),
#          DAYS_FIRST_DUE = ifelse(DAYS_FIRST_DUE == 365243, NA, DAYS_FIRST_DUE),
#          DAYS_LAST_DUE_1ST_VERSION = ifelse(DAYS_LAST_DUE_1ST_VERSION == 365243, NA, DAYS_LAST_DUE_1ST_VERSION),
#          DAYS_LAST_DUE = ifelse(DAYS_LAST_DUE == 365243, NA, DAYS_LAST_DUE),
#          DAYS_TERMINATION = ifelse(DAYS_TERMINATION == 365243, NA, DAYS_TERMINATION),
#          prev_app_credit_ratio = AMT_APPLICATION / AMT_CREDIT,
#          prev_credit_annuity_ratio = AMT_CREDIT / AMT_ANNUITY
#   ) %>%  #new
#   group_by(SK_ID_CURR) %>%
# summarise_all(fn)


rm(prev,fn,x); gc()

 
tri <- 1:nrow(tr)
y <- tr$TARGET

cat("Combining training and test sets...\n")
gc()
tr_te <- tr %>% 
  select(-TARGET) %>% 
  bind_rows(te) %>%
  #left_join(sum_bureau, by = "SK_ID_CURR") %>%
  left_join(sum_bureau_active, by = "SK_ID_CURR") %>%
  left_join(sum_bureau_closed, by = "SK_ID_CURR") %>%
  #left_join(sum_cc_balance, by = "SK_ID_CURR") %>% 
  left_join(cc_3month, by = "SK_ID_CURR") %>%
  left_join(cc_6month, by = "SK_ID_CURR") %>%
  left_join(cc_12month, by = "SK_ID_CURR") %>%
  
  #left_join(sum_payments, by = "SK_ID_CURR") %>%
  left_join(sum_payments_5, by = "SK_ID_CURR") %>%
  #left_join(sum_payments_30, by = "SK_ID_CURR") %>%
  left_join(sum_payments_90, by = "SK_ID_CURR") %>%
  
  #left_join(sum_pc_balance, by = "SK_ID_CURR") %>%
  left_join(pc_3month, by = "SK_ID_CURR") %>%
  left_join(pc_6month, by = "SK_ID_CURR") %>%
  left_join(pc_12month, by = "SK_ID_CURR") %>%
  
  left_join(sum_prev, by = "SK_ID_CURR") 

#######################################################################
#  tr_te <- catto_mean(tr_te,NAME_CONTRACT_TYPE_mean,response = TARGET) x
#  tr_te <- catto_mean(tr_te,NAME_CONTRACT_STATUS_mean,response = TARGET) 
#  tr_te <- catto_mean(tr_te,CODE_REJECT_REASON_mean,response = TARGET)
# 
# tr_te$TARGET <- NULL
######################################################################  

rm(tr,te,active_sum_cc,closed_sum_cc,sum_cc_balance,sum_payments,
   sum_pc_balance,sum_prev, sum_bureau, sum_bureau_active,sum_bureau_closed,sum_active_prev,
   sum_closed_prev,cc_12month,cc_3month,cc_6month, pc_3month, pc_6month, pc_12month,
   sum_payments_30,sum_payments_90,sum_payments_5,sum_prev_walk,sum_prev_x,sum_prev_xna);gc()

tr_te <- tr_te %>%
  select(-SK_ID_CURR) %>%
  mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%   #%>% as.integer
  mutate(
        ANNUITY_LENGTH = AMT_CREDIT / AMT_ANNUITY,
        
        LOAN_INCOME_RATIO = AMT_CREDIT / AMT_INCOME_TOTAL,
        phone_to_employ_ratio = DAYS_LAST_PHONE_CHANGE / DAYS_EMPLOYED,
        LTV = AMT_CREDIT / AMT_GOODS_PRICE,                #cv : 0.7680671 0.7689548
        annuity_length_2.0 = days_employed_by_10,
        birth_car_age_diff = DAYS_BIRTH - OWN_CAR_AGE,
        #####NEW
        term = ifelse(ANNUITY_LENGTH>= 11 & ANNUITY_LENGTH < 16.5, 1, 0), #16.5
        term_2.0 = ifelse(annuity_length_2.0 >= 11 & annuity_length_2.0 < 16.5, 1, 0),
        #term_3.0 = ANNUITY_LENGTH / 21.612,
        
        ######
        row_means = apply(., 1, function(x) mean(x, na.rm = TRUE))
        ) %>%

  mutate_all(funs(ifelse(is.nan(.), NA, .))) %>%
  mutate_all(funs(ifelse(is.infinite(.), NA, .)))

female_no_assets <- tr_te %>%
  select(c(FLAG_OWN_CARN,FLAG_OWN_REALTYN,CODE_GENDERF)) %>%
  group_by(CODE_GENDERF) %>%
  transmute(female_no_car_house = ifelse(CODE_GENDERF == 0,0,FLAG_OWN_CARN + FLAG_OWN_REALTYN))

tr_te$female_no_assets <- female_no_assets$female_no_car_house

male_no_assets <- tr_te %>%
  select(c(FLAG_OWN_CARN,FLAG_OWN_REALTYN,CODE_GENDERM)) %>%
  group_by(CODE_GENDERM) %>%
  transmute(male_no_car_house = ifelse(CODE_GENDERM == 0,0, FLAG_OWN_CARN + FLAG_OWN_REALTYN))

tr_te$male_no_assets <- male_no_assets$male_no_car_house

female_assets <- tr_te %>%
  select(c(FLAG_OWN_CARY,FLAG_OWN_REALTYY,CODE_GENDERF)) %>%
  group_by(CODE_GENDERF) %>%
  transmute(female_assets = ifelse(CODE_GENDERF == 0,0,FLAG_OWN_CARY + FLAG_OWN_REALTYY))
tr_te$female_assets <- female_assets$female_assets


############################################
rm(male_no_assets,female_no_assets,female_assets,male_assets,q,days_employed_by_10,x); gc()

tr_te$TARGET <- NULL

#write_csv(tr_te,"~/Desktop/credit_loan/tr_te.csv")

###Imputing appears to be hurting cv score
#######Impute (median) NAs
med <- apply(tr_te,2,median,na.rm=TRUE)
imputed <-Impute(tr_te, med)

# tr_te$dti = (tr_te$AMT_CREDIT_SUM_DEBT_mean+tr_te$AMT_BALANCE_mean+
#                              tr_te$AMT_INSTALMENT_mean) / tr_te$monthlyIncome

#tr_te <- tr_te %>% group_by(CODE_GENDER, DAYS_BIRTH) %>% mutate(avg_income_by_gender_age = AMT_INCOME_TOTAL/mean(AMT_INCOME_TOTAL))
#tr_te <- tr_te %>% group_by(DAYS_BIRTH) %>% mutate(annuity_length_by_age = AMT_CREDIT / AMT_ANNUITY)

# insig = c('FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_7','FLAG_DOCUMENT_10','FLAG_DOCUMENT_12',
#           'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_17','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20',
#           'FLAG_DOCUMENT_21')
# 
# tr_te = tr_te[!names(tr_te) %in% insig]



# tr_te <- tr_te %>%
#   mutate_if(is.character, as.factor)
# categorical_features <- c("NAME_CONTRACT_TYPE","CODE_GENDER","FLAG_OWN_CAR","FLAG_OWN_REALTY","NAME_TYPE_SUITE",
#                   "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE","NAME_FAMILY_STATUS",
#                   "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE",
#                   "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE")

#female_prob_default = ifelse(CODE_GENDERF == 1,0.57,0.43)
# b<-c(Inf,-12413,-19682,-Inf)
# names <- c("Young", "Middle", "Old")
# tr_te$Age.cat <-cut(tr_te$DAYS_BIRTH, breaks = b, labels = names)
# tr_te$Age.cat <- as.numeric(tr_te$Age.cat)
####################################################################
####################################################################
end_time <- Sys.time()

time_diff_1 <- end_time - start_time
cat("Data Preparation Time Elapsed: ", time_diff_1, "\n")
##################################################################
#annuity_credit_mul = AMT_ANNUITY * AMT_CREDIT,
# goods_age_div = AMT_GOODS_PRICE / DAYS_BIRTH,
# goods_age_mul = AMT_GOODS_PRICE * DAYS_BIRTH,
# goods_employed_div = AMT_GOODS_PRICE / DAYS_EMPLOYED,
# goods_employed_mul = AMT_GOODS_PRICE * DAYS_EMPLOYED,
# birth_employed_div = DAYS_EMPLOYED / DAYS_BIRTH,
#birth_employed_mul = DAYS_EMPLOYED * DAYS_BIRTH,
# employed_credit_max_div = DAYS_EMPLOYED / DAYS_CREDIT_max,
# employed_credit_max_mul = DAYS_EMPLOYED * DAYS_CREDIT_max,
# birth_credit_max_div = DAYS_BIRTH / DAYS_CREDIT_max,
#birth_credit_max_mul = DAYS_BIRTH * DAYS_CREDIT_max,
#edu_type_payment_div = NAME_EDUCATION_TYPE / AMT_PAYMENT_min,
#edu_type_payment_mul = NAME_EDUCATION_TYPE * AMT_PAYMENT_min,
#disposeable_income = AMT_INCOME_TOTAL - (AMT_CREDIT + AMT_ANNUITY)
#na = apply(., 1, function(x) sum(is.na(x))),
#days_employed_debt_ratio = DAYS_EMPLOYED / AMT_CREDIT_SUM_DEBT_max,
#short_employment = ifelse(DAYS_EMPLOYED > -2000 , 1,0),
##cc_debt_prev_credit_ratio = cc_debt_max / AMT_CREDIT_max,
##cc_debt_prev_credit_ratio = cc_debt_sum / AMT_CREDIT_sum,
#age_annuity_length_mul = DAYS_BIRTH * ANNUITY_LENGTH,
#disposable_income = AMT_INCOME_TOTAL-(AMT_CREDIT_SUM_DEBT_mean+AMT_BALANCE_mean+AMT_INSTALMENT_mean),
#balance_income_ratio = AMT_BALANCE_mean / AMT_INCOME_TOTAL,
#children_ratio = CNT_CHILDREN / CNT_FAM_MEMBERS,
#
#income_fam_ratio = AMT_INCOME_TOTAL / CNT_FAM_MEMBERS
#log_debt_cred_ratio = log(debt_credit_ratio_max)
#dti = AMT_CREDIT_SUM_DEBT_max / AMT_INCOME_TOTAL
#days_employed_goods= DAYS_EMPLOYED / AMT_GOODS_PRICE
#birth_employed_ratio = DAYS_BIRTH - DAYS_EMPLOYED
#monthlyIncome = AMT_INCOME_TOTAL / 12
#credit_collateral_ratio = AMT_CREDIT/ (AMT_ANNUITY + AMT_INCOME_TOTAL)
# row_mean = apply(., 1, function(x) mean(x))
#cnt_drawings_by_8 = cnt_drawings_by_8
##doesnt change cv score
# annuity_length_norml = (ANNUITY_LENGTH - mean(ANNUITY_LENGTH)) / sd(ANNUITY_LENGTH),
#exts = (sum(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3) - mean(sum(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3))) / sd(sum(EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3))
# ext3 = EXT_SOURCE_3 / mean(EXT_SOURCE_3),
# annuity_length_sd = sd(ANNUITY_LENGTH),
# ext_prod = ((EXT_SOURCE_1*EXT_SOURCE_2*EXT_SOURCE_3)-mean(EXT_SOURCE_1*EXT_SOURCE_2*EXT_SOURCE_3)) / sd(EXT_SOURCE_1*EXT_SOURCE_2*EXT_SOURCE_3)