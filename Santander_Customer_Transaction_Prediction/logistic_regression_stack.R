tr_te2 <- tr_te

##OOF Predictions
vars <- c(
  "pred4", 
  "pred7",
  "pred_xgb",
  "catboost"
)

tr_te2 <- tr_te[,vars]
tr_te2_test <- tr_te2[-tri,]
tr_te2 <- tr_te2[tri,]

tr_te2$target <- y

#Logistic regression stack
mod <- glm(target ~., data = tr_te2, family = "binomial")
summary(mod)

AUC(y,predict(mod, newdata=tr_te2, type="response"))

# library(MASS)
# logitModelNew <- stepAIC(mod, trace = 1)
# summary(logitModelNew)
# AUC(y,predict(logitModelNew, newdata=tr_te2, type="response"))

pred_stack_tr = predict(mod, newdata = tr_te2, type = "response")
pred_stack = predict(mod, newdata = tr_te2_test, type = "response")

# read_csv("sample_submission.csv") %>%
#   mutate(target = pred_stack) %>%
#   write_csv("logstack_cv-0.9076113_2019-4-10.csv")
#######################################################################
