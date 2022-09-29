library(Matrix)
library(tidyr)
library(ggplot2)
library(lmerTest)

output_N_chart <- function(N_data, dataset_name, N_value) {
  data_plot <- ggplot(N_data, aes(fill=Embedding, y=F1..Partial., x=candidateGeneration.Ranking.Selection)) +
    geom_bar(position='dodge', stat='identity') +
    ggtitle(paste('Embedding Similarity-based Automatic Keyphrase Extraction on', dataset_name, 'Dataset', sep=" ")) +
    xlab('Candidate Generation and Selection Methods') +
    ylab(paste('Partial F1 Score @', N_value, sep=" ")) +
    scale_fill_manual('Embedding Type', values=c('blue4', 'turquoise4', 'green4', 'purple4'))

  return (data_plot)
}


get_data_and_output_chart <- function(dataset_name, dataset_filepath) {
  data <- read.csv(file=dataset_filepath, heade=TRUE, sep= ",")

  data$Ranking.Selection <- with(data, paste(Ranking, Selection, sep="-"))
  data$Ranking.Selection[data$Ranking.Selection == "cos-doc-cand-top-n"] <- "top-n"
  data$Ranking.Selection[data$Ranking.Selection == "cos-doc-cand-top-n-mmr"] <- "mmr"
  data$Ranking.Selection[data$Ranking.Selection == "cos-doc-cand-top-n-mss"] <- "mss"
  data$Ranking.Selection[data$Ranking.Selection == "cos-then-sum-top-n"] <- "top-n-modified"
  data$candidateGeneration.Ranking.Selection <- with(data, paste(Candidate.Keyphrase.Generation, Ranking.Selection, sep="-"))
  data$Embedding.N <- with(data, paste(Embedding, N, sep="-"))
  data <- data[which(data$Algorithm == 'Embedding-based'), ]
  long_data <- subset(data, select=c("Embedding.N", "candidateGeneration.Ranking.Selection", "F1..Partial."))
  long_data_anova <- subset(data, select=c("Dataset", "Candidate.Keyphrase.Generation", "Embedding", "Ranking.Selection", "N", "F1..Partial."))
  # print(head(long_data, 10))
  # print(head(data, 10))
  data_at_N <- subset(data, select=c("Embedding", "N", "candidateGeneration.Ranking.Selection", "F1..Partial."))
  # print(head(data_at_N, 10))
  data_at_1 <- data_at_N[data_at_N$N == 1, ]
  data_at_5 <- data_at_N[data_at_N$N == 5, ]
  data_at_10 <- data_at_N[data_at_N$N == 10, ]

  data_plot_at_1 <- output_N_chart(data_at_1, dataset_name, 1)
  data_plot_at_5 <- output_N_chart(data_at_5, dataset_name, 5)
  data_plot_at_10 <- output_N_chart(data_at_10, dataset_name, 10)

  overview <- ggplot(long_data, aes(fill=Embedding.N, y=F1..Partial., x=candidateGeneration.Ranking.Selection)) +
    geom_bar(position='dodge', stat='identity') +
    ggtitle(paste('Embedding Similarity-based Automatic Keyphrase Extraction on', dataset_name, 'Dataset', sep=" ")) +
    xlab('Candidate Generation and Selection Methods') +
    ylab('Partial F1 Score') +
    scale_fill_manual('Embedding Type @ N', values=c('lightblue', 'blue2', 'blue4', 'turquoise', 'turquoise2', 'turquoise4', 'lightgreen', 'green2', 'green4', 'magenta', 'purple2', 'purple4'))

  outfile <- paste("./study_analysis/", dataset_name, "_plots.pdf", sep="")
  plots <- list(overview, data_plot_at_1, data_plot_at_5, data_plot_at_10)
  # ggsave(outfile,
     # arrangeGrob(grobs=plots),
     # device = cairo_pdf,
     # width = 297,
     # height = 210,
     # units = "mm")
  pdf(outfile, width=14, height=8.5, onefile=TRUE)
  # plots
  for (plot in plots) {
    print(plot)
  }
  # dev.off()
  return (long_data_anova)
}

inspec_path <- '/home/jwood/OneDrive/SMART/Jesse_2022/skill_labeling/ake_study_embedding_2_24_2022/inspec_evaluation.csv'
kdd_path <- '/home/jwood/OneDrive/SMART/Jesse_2022/skill_labeling/ake_study_embedding_2_24_2022/kdd_evaluation.csv'
oli_gen_chem_path <- '/home/jwood/OneDrive/SMART/Jesse_2022/skill_labeling/ake_study_embedding_2_24_2022/oli-gen-chem_evaluation.csv'
oli_intro_bio_path <- '/home/jwood/OneDrive/SMART/Jesse_2022/skill_labeling/ake_study_embedding_2_24_2022/oli-intro-bio_evaluation.csv'

inspec_anova_data <- get_data_and_output_chart('Inspec', inspec_path)
kdd_anova_data <- get_data_and_output_chart('KDD', kdd_path)
# chem_short_data <- get_data_and_output_chart('OLI General Chemistry 1 (Short Form)', oli_gen_chem_path_short)
chem_data <- get_data_and_output_chart('OLI General Chemistry 1', oli_gen_chem_path)
bio_data <- get_data_and_output_chart('OLI Introduction to Biology', oli_intro_bio_path)
anova_data <- rbind(inspec_anova_data, kdd_anova_data)
anova_data <- rbind(anova_data, chem_data)
anova_data <- rbind(anova_data, bio_data)

# train model
print(paste('Training Model for Regression Test and ANOVA: ', Sys.time()))
anova_data$Dataset <- factor(anova_data$Dataset, levels=c("inspec", "kdd", "oli-gen-chem", "oli-intro-bio"), ordered=FALSE)
anova_data$Candidate.Keyphrase.Generation <- factor(anova_data$Candidate.Keyphrase.Generation, levels=c("parsing", "ngrams"), ordered=FALSE)
anova_data$Embedding <- factor(anova_data$Embedding, levels=c("sbert-mean-pooling", "w2v", "glove", "d2v"), ordered=FALSE)
anova_data$Ranking.Selection <- factor(anova_data$Ranking.Selection, levels=c("top-n", "top-n-modified", "mmr", "mss"), ordered=FALSE)

anova_data$N <- as.numeric(as.vector(anova_data$N))
anova_data$F1..Partial. <- as.numeric(as.vector(anova_data$F1..Partial.))

model <- lmer(F1..Partial. ~ Candidate.Keyphrase.Generation + Embedding + Ranking.Selection + (1|N) + (1|Dataset), data=anova_data)
summary(model)
anova(model)
