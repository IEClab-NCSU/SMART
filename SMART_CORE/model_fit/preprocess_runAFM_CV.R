library(lme4)
# Attach packages
library(cvms)
library(groupdata2) # fold()
library(dplyr) # %>% arrange()

### load dataset (student step roll-up)
# filename='output/run_1/assessment/tfidf/second/50/1_assessment_skill_second_tfidf_50_assessment_StudentStep_KC_Opportunity.csv'
# filename='/home/iec/Raj/smart_outputs/03_24_2021_bert_clusterid_npPreprocessing/run_1/assessment/bert/first/50/1_assessment_skill_first_bert_50_assessment_StudentStep_KC_Opportunity.csv'
filename='output/run_1/assessment/tf/first/50/1_assessment_skill_first_tf_50_assessment_StudentStep_KC_Opportunity.csv'
#origRollup <- data.frame(read.table(file="ds76.txt",na.string="NA",sep="\t",quote="",header=T))
# args <- commandArgs(trailingOnly = TRUE)
# filename <- args[1]
origRollup <- read.csv(file = filename, header = TRUE, sep = ",", dec = ".")

### extract relevant columns from student-step rollup and rename to 

KC_model_index <- grep('KC_SMART', names(origRollup))  ## input KC/skill model name here
df <- origRollup[,c(3,5,7,15,KC_model_index[1],KC_model_index[1]+1)]  ## subset only the columns of interest

#########
# df = df_full[1:50000,]
#########

# shuffling rows
# set.seed(42)
# rows <- sample(nrow(df))
# df <- df[rows, ]


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

# Fold data
data <- fold(
  df,
  k = 4,
  cat_col = "Success",
) %>%
  arrange(.folds)

print('Cross validating..')

cv <- cross_validate(
  data,
  formulas = "Success ~ KC + Opportunity + Opportunity:KC + (Opportunity|Anon.Student.Id)",
  family="binomial",
  fold_cols = ".folds",
  REML = FALSE,
  parallel=TRUE
)

print(cv)