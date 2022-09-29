library(Matrix)
library(tidyr)
library(lmerTest)

# load the ami score data from file
filename <- 'hyperparameter_selection_model_fit_data_long.csv'
data <- read.csv(file = filename, header = TRUE, sep = ",")

# train model
print(paste('Training Model for Regression Test and ANOVA: ', Sys.time()))
data$representation_level <- factor(data$representation_level, levels=c("first", "second"), ordered=FALSE)
data$course <- factor(data$course, levels=c("oli_intro_bio", "oli_gen_chem"), ordered=FALSE)
data$focal_strategy <- factor(data$focal_strategy, levels=c("assessment", "paragraph"), ordered=FALSE)
data$k <- as.numeric(as.vector(data$num_clusters))
data$aic <- as.numeric(as.vector(data$aic))

# run regression test and ANOVA
model <- lmer(aic ~ representation_level + focal_strategy + k + (1|course), data=data)
summary(model)
anova(model)

