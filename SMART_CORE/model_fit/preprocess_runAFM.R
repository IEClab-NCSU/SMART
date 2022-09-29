library(lme4)

#setwd("~/Box Sync/Research/DATA/")  ## directory where your step roll-up dataset is
args <- commandArgs(trailingOnly = TRUE)
filename <- args[1]
#filename='StudentStepRollUp_KC_local_reduced.csv'

### load dataset (student step roll-up)
#origRollup <- data.frame(read.table(file="ds76.txt",na.string="NA",sep="\t",quote="",header=T))
origRollup <- read.csv(file = filename, header = TRUE, sep = ",", dec = ".")

### extract relevant columns from student-step rollup and rename to 

KC_model_index <- grep('KC_SMART', names(origRollup))  ## input KC/skill model name here
df <- origRollup[,c(3,5,7,15,KC_model_index[1],KC_model_index[1]+1)]  ## subset only the columns of interest

#########
# df = df[1:1000,]
#########

df$First.Attempt <- gsub("incorrect", 0, df$First.Attempt)  ## convert correctness coding to binary, numeric
df$First.Attempt <- gsub("hint", 0, df$First.Attempt)
df$First.Attempt <- gsub("correct", 1, df$First.Attempt)

names(df) <- c("Anon.Student.Id","Problem.Name","Step.Name","Success","KC","Opportunity")  ## rename columns
df$Success <- as.numeric(as.vector(df$Success))  ## convert success and opportunity columns to numeric
df$Opportunity <- as.numeric(as.vector(df$Opportunity)) - 1

#str(df)  ## view summary of columns
# summary(df)

# print('Fitting model...')

#current (12_16_2020)
### run a mixed-model AFM - with fixed KC intercept, fixed slope; random student intercept, random studeent slope, and random KC by student slope
# afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (1+Opportunity|Anon.Student.Id) + (Opportunity|Anon.Student.Id:KC), data=df, family=binomial())

#Dropping a tern in the above equation for simpler model
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id), data=df, family=binomial())

# DataShop: https://pslcdatashop.web.cmu.edu/help?page=rSoftware
# afm.model.st <- glmer(Success ~ KC + KC:Opportunity+(1|Anon.Student.Id),data=df,family=binomial())

# previous (before 07_27_2020)
# ### run regular AFM - with student intercept, KC intercept, and KC slope as random effects
# afm.model.st <- glmer(Success ~ (1|Anon.Student.Id) + (Opportunity|KC) - 1,
#                        data=df, family=binomial(), nAGQ = 0, verbose=1)

AIC(afm.model.st)  ## output AIC
BIC(afm.model.st)  ## output BIC
nobs(afm.model.st) # total observations
logLik(afm.model.st) # log-likelihood, total parameters (degree of freedom)

# afm.model.st.2 <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id:KC), data=df, family=binomial())

# anova(afm.model.st, afm.model.st.2)

df$Prediction <- predict(afm.model.st, newdata = df, type = "response")  ## fill a column with AFM's predicted performance
ranef(afm.model.st)  ## view random-effects coefficient estimates




# ### run regular AFM - with student intercept as a random effect, KC intercept/slope as fixed effects
# afm.model.reg <- glmer(Success ~ (1|Anon.Student.Id) + KC + KC:Opportunity - 1,
#                        data=df, family=binomial())
# AIC(afm.model.reg)  ## output AIC
# BIC(afm.model.reg)  ## output BIC
# coef(afm.model.reg)
# df$Prediction <- predict(afm.model.reg, newdata = df, type = "response")  ## fill a column with AFM's predicted performance

