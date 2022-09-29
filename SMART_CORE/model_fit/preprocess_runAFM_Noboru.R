library(lme4)

# setwd("~/Box Sync/Research/DATA/")  ## directory where your step roll-up dataset is

### load dataset (student step roll-up)
origRollup <- data.frame(read.table(file="ds445.txt",na.string="NA",sep="\t",quote="",header=T))

### extract relevant columns from student-step rollup and rename to 

KC_model_index <- grep('KC..BenchUnit.', names(origRollup))  ## input KC/skill model name here
df <- origRollup[,c(3,5,7,15,KC_model_index[1],KC_model_index[1]+1)]  ## subset only the columns of interest

df$First.Attempt <- gsub("incorrect", 0, df$First.Attempt)  ## convert correctness coding to binary, numeric
df$First.Attempt <- gsub("hint", 0, df$First.Attempt)
df$First.Attempt <- gsub("correct", 1, df$First.Attempt)

names(df) <- c("Anon.Student.Id","Problem.Name","Step.Name","Success","KC","Opportunity")  ## rename columns
df$Success <- as.numeric(as.vector(df$Success))  ## convert success and opportunity columns to numeric
df$Opportunity <- as.numeric(as.vector(df$Opportunity)) - 1

str(df)  ## view summary of columns
# 'data.frame':	4349 obs. of  6 variables:
 # $ Anon.Student.Id: Factor w/ 51 levels "Stu_00534e69904177f44e6e707bdd26d217",..: 1 1 1 1 1 1 1 1 1 1 ...
 # $ Problem.Name   : Factor w/ 33 levels "1","1/10","1/2",..: 23 31 30 21 26 1 2 22 25 20 ...
 # $ Step.Name      : Factor w/ 55 levels "1/10Click0-1",..: 43 55 50 38 44 19 1 36 41 34 ...
 # $ Success        : num  0 1 1 0 1 1 0 0 0 0 ...
 # $ KC             : Factor w/ 7 levels "","Half","NonUnitNearBench",..: 7 7 7 7 7 7 5 4 4 4 ...
 # $ Opportunity    : num  0 1 2 3 4 5 0 0 1 2 ...


# Original Ran Liu's model, which looks odd to me: (1) KC is a fixed effect, (2) KC shuold have intercept 
#
# iafm.model.reg <- glmer(Success ~ (1|Anon.Student.Id) + (Opportunity|Anon.Student.Id) + (Opportunity|KC) - 1,
#                       data=df, family=binomial())


### run a mixed-model AFM - with fixed KC intercept, fixed slope; random student intercept, random studeent slope, and random KC by student slope
afm.model.st <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (1+Opportunity|Anon.Student.Id) + (Opportunity|Anon.Student.Id:KC), data=df, family=binomial())
AIC(afm.model.st)  ## output AIC
# [1] 5107.934
BIC(afm.model.st)  ## output BIC
# [1] 5222.641

#
# Adding Student by KC random slope improved the model with only student random slope 
#
afm.model.st.noKC <- glmer(Success ~ KC + Opportunity + Opportunity:KC + (1+Opportunity|Anon.Student.Id), data=df, family=binomial())

anova(afm.model.st.noKC, afm.model.st)
# Data: df
# Models:
# afm.model.st.noKC: Success ~ KC + Opportunity + Opportunity:KC + (1 + Opportunity | 
# afm.model.st.noKC:     Anon.Student.Id)
# afm.model.st: Success ~ KC + Opportunity + Opportunity:KC + (1 + Opportunity | 
# afm.model.st:     Anon.Student.Id) + (Opportunity | Anon.Student.Id:KC)
                  # npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)    
# afm.model.st.noKC   15 5127.7 5223.3 -2548.9   5097.7                         
# afm.model.st        18 5107.9 5222.6 -2536.0   5071.9 25.786  3  1.057e-05 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#
# Addinng student as random factor improved the model
#
afm.model <- glm(Success ~ KC + Opportunity + Opportunity:KC, data=df, family=binomial())
AIC(afm.model)
# [1] 5831.302
BIC(afm.model)
# [1] 5907.774

anova(afm.model.st, afm.model)
# Data: df
# Models:
# afm.model: Success ~ KC + Opportunity + Opportunity:KC
# afm.model.st: Success ~ KC + Opportunity + Opportunity:KC + (1 + Opportunity | 
# afm.model.st:     Anon.Student.Id) + (Opportunity | Anon.Student.Id:KC)
             # npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)    
# afm.model      12 5831.3 5907.8 -2903.7   5807.3                         
# afm.model.st   18 5107.9 5222.6 -2536.0   5071.9 735.37  6  < 2.2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


df$Prediction <- predict(afm.model, newdata = df, type = "response")  ## fill a column with AFM's predicted performance
ranef(afm.model)  ## view random-effects coefficient estimates


### run regular AFM - with student intercept, KC intercept, and KC slope as random effects
afm.model.reg <- glmer(Success ~ (1|Anon.Student.Id) + (Opportunity|KC) - 1, data=df, family=binomial())
AIC(afm.model.reg)  ## output AIC
# [1] 5156.103
BIC(afm.model.reg)  ## output BIC
# [1] 5187.966

anova(afm.model, afm.model.reg)
# Data: df
# Models:
# afm.model.reg: Success ~ (1 | Anon.Student.Id) + (Opportunity | KC) - 1
# afm.model: Success ~ KC + Opportunity + Opportunity:KC + (1 + Opportunity | 
# afm.model:     Anon.Student.Id) + (Opportunity | Anon.Student.Id:KC)
              # npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)    
# afm.model.reg    5 5156.1 5188.0 -2573.1   5146.1                         
# afm.model       18 5107.9 5222.6 -2536.0   5071.9 74.169 13  1.359e-10 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

df$Prediction <- predict(afm.model.reg, newdata = df, type = "response")  ## fill a column with AFM's predicted performance
ranef(afm.model.reg)  ## view random-effects coefficient estimates

#
# The original model on DataShop 
# https://pslcdatashop.web.cmu.edu/ExternalTools?toolId=11
#
### run regular AFM - with student intercept as a random effect, KC intercept/slope as fixed effects
afm.model.org <- glmer(Success ~ (1|Anon.Student.Id) + KC + KC:Opportunity - 1, data=df, family=binomial())
AIC(afm.model.st)  ## output AIC
BIC(afm.model.st)  ## output BIC

anova(afm.model.org, afm.model.st)
# Data: df
# Models:
# afm.model.org: Success ~ (1 | Anon.Student.Id) + KC + KC:Opportunity - 1
# afm.model.st: Success ~ KC + Opportunity + Opportunity:KC + (1 + Opportunity | 
# afm.model.st:     Anon.Student.Id) + (Opportunity | Anon.Student.Id:KC)
              # npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)    
# afm.model.org   13 5130.7 5213.5 -2552.3   5104.7                         
# afm.model.st    18 5107.9 5222.6 -2536.0   5071.9 32.766  5  4.188e-06 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


coef(afm.model.reg)
df$Prediction <- predict(afm.model.reg, newdata = df, type = "response")  ## fill a column with AFM's predicted performance
