# Load the necessary libraries
library(readxl)
library(dplyr)
library(broom)
library(effsize)
library(boot)

# Define the necessary variables
timepoint_1 <- "before"
timepoint_2 <- "after"
treatment_1 <- "Ayahuasca"
treatment_2 <- "Placebo"
group_1 <- "PG"
group_2 <- "CG"
biomarkers <- c("ldlcholesterol", "glucose", "crp", "ast", "bdnf", "rbc", "leukocytes","neutrophils","neutrophils2", "segmented","segmented2", "eosinophils", "eosinophils2", "lymphocyte", "lymphocyte2", "monocytes", "monocytes2"
)

# Load the data
h6dt <- read_excel("h6dt.xlsx")

# Create the list of filtered dataframes for each biomarker
df_lists <- lapply(biomarkers, function(biomarker) {
  df1 <- h6dt %>% filter(timepoint == timepoint_1, treatment == treatment_1, group == group_1) %>% select(subjectid, biomarker)
  df2 <- h6dt %>% filter(timepoint == timepoint_2, treatment == treatment_1, group == group_1) %>% select(subjectid, biomarker)
  df3 <- h6dt %>% filter(timepoint == timepoint_1, treatment == treatment_2, group == group_1) %>% select(subjectid, biomarker)
  df4 <- h6dt %>% filter(timepoint == timepoint_2, treatment == treatment_2, group == group_1) %>% select(subjectid, biomarker)
  df5 <- h6dt %>% filter(timepoint == timepoint_1, treatment == treatment_1, group == group_2) %>% select(subjectid, biomarker)
  df6 <- h6dt %>% filter(timepoint == timepoint_2, treatment == treatment_1, group == group_2) %>% select(subjectid, biomarker)
  df7 <- h6dt %>% filter(timepoint == timepoint_1, treatment == treatment_2, group == group_2) %>% select(subjectid, biomarker)
  df8 <- h6dt %>% filter(timepoint == timepoint_2, treatment == treatment_2, group == group_2) %>% select(subjectid, biomarker)
  
  return(list(df1, df2, df3, df4, df5, df6, df7, df8))
})

# Define the function to perform the Wilcoxon test on each pair of dataframes
perform_wilcoxon_test <- function(df_list, biomarker) {
  df1 <- df_list[[1]]
  df2 <- df_list[[2]]
  df3 <- df_list[[3]]
  df4 <- df_list[[4]]
  df5 <- df_list[[5]]
  df6 <- df_list[[6]]
  df7 <- df_list[[7]]
  df8 <- df_list[[8]]
  
  test1 <- wilcox.test(df1[[2]], df2[[2]], paired = TRUE)
  test2 <- wilcox.test(df3[[2]], df4[[2]], paired = TRUE)
  test3 <- wilcox.test(df5[[2]], df6[[2]], paired = TRUE)
  test4 <- wilcox.test(df7[[2]], df8[[2]], paired = TRUE)
  
  # Calculate effect size (Cohen's d)
  effect_size1 <- cohen.d(df1[[2]], df2[[2]], paired = TRUE)
  effect_size2 <- cohen.d(df3[[2]], df4[[2]], paired = TRUE)
  effect_size3 <- cohen.d(df5[[2]], df6[[2]], paired = TRUE)
  effect_size4 <- cohen.d(df7[[2]], df8[[2]], paired = TRUE)
  
  # Calculate confidence intervals using bootstrapping
  bootstrap_ci <- function(data1, data2) {
    boot_func <- function(data, indices) {
      d <- cohen.d(data[indices], data2[indices], paired = TRUE)
      return(d$estimate)
    }
    bootstrap <- boot(data1, boot_func, R = 1000)
    ci <- boot.ci(bootstrap, type = "bca")
    return(ci$bca[4:5])
  }
  
  ci1 <- bootstrap_ci(df1[[2]], df2[[2]])
  ci2 <- bootstrap_ci(df3[[2]], df4[[2]])
  ci3 <- bootstrap_ci(df5[[2]], df6[[2]])
  ci4 <- bootstrap_ci(df7[[2]], df8[[2]])
  
  results <- data.frame(
    Variable = biomarker,
    Group = c(paste0("Ayahuasca_", group_1), paste0("Placebo_", group_1), paste0("Ayahuasca_", group_2), paste0("Placebo_", group_2)),
    Median_Before = c(
      median(df1[[2]], na.rm = TRUE),
      median(df3[[2]], na.rm = TRUE),
      median(df5[[2]], na.rm = TRUE),
      median(df7[[2]], na.rm = TRUE)
    ),
    Median_After = c(
      median(df2[[2]], na.rm = TRUE),
      median(df4[[2]], na.rm = TRUE),
      median(df6[[2]], na.rm = TRUE),
      median(df8[[2]], na.rm = TRUE)
    ),
    IQR_Before = c(
      IQR(df1[[2]], na.rm = TRUE),
      IQR(df3[[2]], na.rm = TRUE),
      IQR(df5[[2]], na.rm = TRUE),
      IQR(df7[[2]], na.rm = TRUE)
    ),
    IQR_After = c(
      IQR(df2[[2]], na.rm = TRUE),
      IQR(df4[[2]], na.rm = TRUE),
      IQR(df6[[2]], na.rm = TRUE),
      IQR(df8[[2]], na.rm = TRUE)
    ),
    Mean = c(
      round(mean(df1[[2]], na.rm = TRUE), digits = 2),
      round(mean(df2[[2]], na.rm = TRUE), digits = 2),
      round(mean(df3[[2]], na.rm = TRUE), digits = 2),
      round(mean(df4[[2]], na.rm = TRUE), digits = 2)
    ),
    p_value = c(
      format(test1$p.value, digits = ifelse(test1$p.value < 0.01, 3, 4)),
      format(test2$p.value, digits = ifelse(test2$p.value < 0.01, 3, 4)),
      format(test3$p.value, digits = ifelse(test3$p.value < 0.01, 3, 4)),
      format(test4$p.value, digits = ifelse(test4$p.value < 0.01, 3, 4))
    ),
    Effect_Size = c(
      round(effect_size1$estimate, digits = 2),
      round(effect_size2$estimate, digits = 2),
      round(effect_size3$estimate, digits = 2),
      round(effect_size4$estimate, digits = 2)
    ),
    ConfidenceInterval = c(
      paste0("[", round(ci1[1], digits = 2), ", ", round(ci1[2], digits = 2), "]"),
      paste0("[", round(ci2[1], digits = 2), ", ", round(ci2[2], digits = 2), "]"),
      paste0("[", round(ci3[1], digits = 2), ", ", round(ci3[2], digits = 2), "]"),
      paste0("[", round(ci4[1], digits = 2), ", ", round(ci4[2], digits = 2), "]")
    )
  )
  
  return(results)
}

# Create a new dataframe to store all results
results_all <- data.frame()

# Loop through the list of biomarker dataframes
for (i in 1:length(biomarkers)) {
  df_list <- df_lists[[i]]
  results <- perform_wilcoxon_test(df_list, biomarkers[i])
  results_all <- bind_rows(results_all, results)
}

# Arrange the results table
results_all <- results_all[, c("Variable", "Group", "Median_Before", "Median_After", "IQR_Before", "IQR_After", "Mean", "p_value", "Effect_Size", "ConfidenceInterval")]
results_all <- results_all[order(as.numeric(gsub("[^.0-9]", "", results_all$p_value))), ]

# Create separate dataframes for PG and CG results
results_pg <- subset(results_all, grepl("PG", Group))
results_cg <- subset(results_all, grepl("CG", Group))

# Arrange the results tables
results_pg <- results_pg[order(results_pg$Variable), ]
results_cg <- results_cg[order(results_cg$Variable), ]

# Save the results as separate CSV files
write.csv(results_pg, file = "Results_PG.csv", row.names = FALSE)
write.csv(results_cg, file = "Results_CG.csv", row.names = FALSE)
