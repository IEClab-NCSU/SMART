library(lme4)
library(Metrics) 

### load dataset (student step roll-up)
# filename='output/run_1/assessment/tfidf/second/50/1_assessment_skill_second_tfidf_50_assessment_StudentStep_KC_Opportunity.csv'
# filename='/home/iec/Raj/smart_outputs/03_24_2021_bert_clusterid_npPreprocessing/run_1/assessment/bert/first/50/1_assessment_skill_first_bert_50_assessment_StudentStep_KC_Opportunity.csv'
# filename='output/_assessment_skill_second_bert_50_assessment_StudentStep_KC_Opportunity.csv'
#origRollup <- data.frame(read.table(file="ds76.txt",na.string="NA",sep="\t",quote="",header=T))
args <- commandArgs(trailingOnly = TRUE)
filename <- args[1]
origRollup <- read.csv(file = filename, header = TRUE, sep = ",", dec = ".")

### extract relevant columns from student-step rollup and rename to 

KC_model_index <- grep('KC_SMART', names(origRollup))  ## input KC/skill model name here
df <- origRollup[,c(3,5,7,15,KC_model_index[1],KC_model_index[1]+1)]  ## subset only the columns of interest

#########
# df = df_full[1:50000,]
#########

# shuffling rows
set.seed(42)
rows <- sample(nrow(df))
df <- df[rows, ]


df$First.Attempt <- gsub("incorrect", 0, df$First.Attempt)  ## convert correctness coding to binary, numeric
df$First.Attempt <- gsub("hint", 0, df$First.Attempt)
df$First.Attempt <- gsub("correct", 1, df$First.Attempt)

names(df) <- c("Anon.Student.Id","Problem.Name","Step.Name","Success","KC","Opportunity")  ## rename columns
df$Success <- as.numeric(as.vector(df$Success))  ## convert success and opportunity columns to numeric
df$Opportunity <- as.numeric(as.vector(df$Opportunity)) - 1

# smp_size <- floor(0.75 * nrow(df))
# ## set the seed to make your partition reproducible
# set.seed(123)
# train_ind <- sample(seq_len(nrow(df)), size = smp_size)
# 
# train <- df[train_ind, ]
# test <- df[-train_ind, ]

# train <- df[1:1500,]
# test <- df[1501:2000,]



rmse_unblocked <- 0
n_folds <- 5
fold_size <- floor(nrow(df)/n_folds)

df1 <- df[1:fold_size, ]
df2 <- df[(fold_size+1):(fold_size*2), ]
df3 <- df[(fold_size*2+1):(fold_size*3), ]
df4 <- df[(fold_size*3+1):(fold_size*4), ]
df5 <- df[(fold_size*4+1):(fold_size*5), ]


# 1
df_train <- rbind(df2, df3, df4, df5)
df_test <- df1

#Dropping a tern in the above equation for simpler model
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df_train, family=binomial())

predictions <- predict(afm.model.st, newdata = df_test)
rmse_value = rmse(df_test$Success, predictions)
print(rmse_value)
rmse_unblocked <- rmse_unblocked + rmse_value

# 2
df_train <- rbind(df1, df3, df4, df5)
df_test <- df2

#Dropping a tern in the above equation for simpler model
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df_train, family=binomial())

predictions <- predict(afm.model.st, newdata = df_test)
rmse_value = rmse(df_test$Success, predictions)
print(rmse_value)
rmse_unblocked <- rmse_unblocked + rmse_value

# 3
df_train <- rbind(df1, df2, df4, df5)
df_test <- df3

#Dropping a tern in the above equation for simpler model
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df_train, family=binomial())

predictions <- predict(afm.model.st, newdata = df_test)
rmse_value = rmse(df_test$Success, predictions)
print(rmse_value)
rmse_unblocked <- rmse_unblocked + rmse_value

# 4
df_train <- rbind(df1, df2, df3, df5)
df_test <- df4

#Dropping a tern in the above equation for simpler model
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df_train, family=binomial())

predictions <- predict(afm.model.st, newdata = df_test)
rmse_value = rmse(df_test$Success, predictions)
print(rmse_value)
rmse_unblocked <- rmse_unblocked + rmse_value

# 5
df_train <- rbind(df1, df2, df3, df4)
df_test <- df5

#Dropping a tern in the above equation for simpler model
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df_train, family=binomial())

predictions <- predict(afm.model.st, newdata = df_test)
rmse_value = rmse(df_test$Success, predictions)
print(rmse_value)
rmse_unblocked <- rmse_unblocked + rmse_value




#Computing average of rmse
rmse_unblocked <- rmse_unblocked/n_folds

#Dropping a tern in the above equation for simpler model
# afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df, family=binomial())

print(rmse_unblocked) #Cross validation RMSE
# AIC(afm.model.st)  ## output AIC
# BIC(afm.model.st)  ## output BIC
# nobs(afm.model.st) # total observations
# logLik(afm.model.st) # log-likelihood, total parameters (degree of freedom)