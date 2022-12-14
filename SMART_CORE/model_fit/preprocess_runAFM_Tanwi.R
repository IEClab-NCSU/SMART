# This file is used by comparison.sh internally for generating AIC and BIC values. Parameters to be given to this file are the filename generated by SMART in txt format
library(lme4)

#setwd("~/Box Sync/Research/DATA/")  ## directory where your step roll-up dataset is

### load dataset (student step roll-up)
args <- commandArgs(trailingOnly = TRUE)
filename <- args[1]
#origRollup <- data.frame(read.table(file=filename,na.string="NA",sep="\t",quote="",header=T)) #Raj replaced this with the line below
origRollup <- read.csv(file = filename) 


### extract relevant columns from student-step rollup and rename to 

KC_model_index <- grep('KC(SMART)', names(origRollup))  ## input KC/skill model name here
df <- origRollup[,c(2,8,10,13,16,11)]  ## subset only the columns of interest
print (head(df))
df$Outcome <- gsub("INCORRECT", 0, df$Outcome)  ## convert correctness coding to binary, numeric
df$Outcome <- gsub("HINT", 0, df$Outcome)
df$Outcome <- gsub("CORRECT", 1, df$Outcome)

names(df) <- c("Anon.Student.Id","Problem.Name","Step.Name","Success","KC","Opportunity")  ## rename columns
df$Success <- as.numeric(as.vector(df$Success))  ## convert success and opportunity columns to numeric
df$Opportunity <- as.numeric(as.vector(df$Opportunity)) - 1

str(df)  ## view summary of columns


### run regular AFM - with student intercept, KC intercept, and KC slope as random effects
afm.model.reg <- glmer(Success ~ (1|Anon.Student.Id) + (Opportunity|KC) - 1,
                       data=df, family=binomial(), nAGQ = 0, verbose=1)
AIC(afm.model.reg)  ## output AIC
BIC(afm.model.reg)  ## output BIC
df$Prediction <- predict(afm.model.reg, newdata = df, type = "response", allow.new.levels = TRUE)  ## fill a column with AFM's predicted performance
ranef(afm.model.reg)  ## view random-effects coefficient estimates


### run regular AFM - with student intercept as a random effect, KC intercept/slope as fixed effects
afm.model.reg <- glmer(Success ~ (1|Anon.Student.Id) + KC + KC:Opportunity - 1,
                       data=df, family=binomial(), nAGQ = 0, verbose=1)
AIC(afm.model.reg)  ## output AIC
BIC(afm.model.reg)  ## output BIC
coef(afm.model.reg)
df$Prediction <- predict(afm.model.reg, newdata = df, type = "response", allow.new.levels = TRUE)  ## fill a column with AFM's predicted performance
