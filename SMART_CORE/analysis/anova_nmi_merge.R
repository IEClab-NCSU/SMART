library(Matrix)
library(tidyr)
library(lmerTest)

# load the nmi score data from file
filename <- './nmi_study/nmi_scores.csv'
origRollup <- read.csv(file = filename, header = TRUE, sep = ",")

# convert wide data to long data
data_long <- gather(origRollup, trial, nmi_score, nmi_1:nmi_25, factor_key=TRUE)
data_long <- data_long[,c(1,3:4,8)]
print(head(data_long, 10))

# train model
print(paste('Training Model for Regression Test and ANOVA: ', Sys.time()))
data_long$merge_operation <- factor(data_long$merge_operation, levels=c("merge", "no_merge"), ordered=FALSE)
data_long$course <- factor(data_long$course, levels=c("oli_biology", "oli_chemistry"), ordered=FALSE)
data_long$nmi_score <- as.numeric(as.vector(data_long$nmi_score))
data_long$num_clusters <- as.numeric(as.vector(data_long$num_clusters))

model <- lmer(nmi_score ~ merge_operation + num_clusters + (1|course), data=data_long)
summary(model)
anova(model)