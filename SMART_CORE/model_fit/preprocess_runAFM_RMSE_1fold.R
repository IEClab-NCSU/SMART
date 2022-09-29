library(lme4)
library(Metrics) 

### load Student step roll-up dataset
# filename='output/run_1/assessment/tfidf/second/50/1_assessment_skill_second_tfidf_50_assessment_StudentStep_KC_Opportunity.csv'
print(paste('Reading from file for model fit: ', Sys.time()))

args <- commandArgs(trailingOnly = TRUE)
filename <- args[1]
origRollup <- read.csv(file = filename, header = TRUE, sep = ",", dec = ".")

print(paste('Processing Student Response Data for model fit: ', Sys.time()))
### extract relevant columns from student-step rollup and rename to
KC_model_index <- grep('KC_SMART', names(origRollup))  ## input KC/skill model name here
df <- origRollup[,c(3,5,7,15,KC_model_index[1],KC_model_index[1]+1)]  ## subset only the columns of interest

df$First.Attempt <- gsub("incorrect", 0, df$First.Attempt)  ## convert correctness coding to binary, numeric
df$First.Attempt <- gsub("hint", 0, df$First.Attempt)
df$First.Attempt <- gsub("correct", 1, df$First.Attempt)

names(df) <- c("Anon.Student.Id","Problem.Name","Step.Name","Success","KC","Opportunity")  ## rename columns
df$Success <- as.numeric(as.vector(df$Success))  ## convert success and opportunity columns to numeric
df$Opportunity <- as.numeric(as.vector(df$Opportunity)) - 1

# shuffling rows
set.seed(123)
rows <- sample(nrow(df))
df <- df[rows, ]

n_folds <- 5
fold_size <- floor(nrow(df)/n_folds)

df_train <- df[1:(fold_size*4), ]
df_test <- df[(fold_size*4+1):(fold_size*5), ]

df_test <- df_test[df_test$KC %in% df_train$KC,]

print(paste('Training AFM Model: ', Sys.time()))
# AFM model (Dropped a term in the equation for simpler model)
afm.model.st <- glmer(Success ~ KC + KC:Opportunity + (1|Anon.Student.Id), data=df_train, family=binomial())

print(paste('Evaluating AFM Model: ', Sys.time()))
predictions <- predict(afm.model.st, newdata = df_test, type="response")
rmse_unblocked <- rmse(df_test$Success, predictions)

print(rmse_unblocked) #Cross validation RMSE
AIC(afm.model.st)  ## output AIC
BIC(afm.model.st)  ## output BIC
nobs(afm.model.st) # total observations
logLik(afm.model.st) # log-likelihood, total parameters (degree of freedom)