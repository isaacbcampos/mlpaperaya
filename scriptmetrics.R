library(tidyverse)

# Create a data frame with the metric values
data <- tribble(
  ~group, ~condition, ~metric, ~value,
  "(a) Healthy Controls vs. Patients: Baseline", "All variables", "Accuracy", 0.76,
  "(a) Healthy Controls vs. Patients: Baseline", "All variables", "Sensitivity", 0.88,
  "(a) Healthy Controls vs. Patients: Baseline", "All variables", "Specificity", 0.57,
  "(a) Healthy Controls vs. Patients: Baseline", "All variables", "AUROC", 0.60,
  "(a) Healthy Controls vs. Patients: Baseline", "Selected variables", "Accuracy", 0.79,
  "(a) Healthy Controls vs. Patients: Baseline", "Selected variables", "Sensitivity", 0.81,
  "(a) Healthy Controls vs. Patients: Baseline", "Selected variables", "Specificity", 0.75,
  "(a) Healthy Controls vs. Patients: Baseline", "Selected variables", "AUROC", 0.65,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "All variables", "Accuracy", 0.57,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "All variables", "Sensitivity", 0.60,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "All variables", "Specificity", 0.57,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "All variables", "AUROC", 0.71,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "Selected variables", "Accuracy", 0.76,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "Selected variables", "Sensitivity", 0.73,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "Selected variables", "Specificity", 0.80,
  "(c) Ayahuasca vs. Placebo: TRD Patients", "Selected variables", "AUROC", 0.80,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "All variables", "Accuracy", 0.58,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "All variables", "Sensitivity", 0.50,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "All variables", "Specificity", 0.54,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "All variables", "AUROC", 0.56,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "Selected variables", "Accuracy", 0.63,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "Selected variables", "Sensitivity", 0.45,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "Selected variables", "Specificity", 0.54,
  "(b) Ayahuasca vs. Placebo: Healthy Controls", "Selected variables", "AUROC", 0.53
)

# Remove rows with missing values
data <- drop_na(data)

# Set the order of the groups and conditions
data$group <- factor(data$group, levels = c("(a) Healthy Controls vs. Patients: Baseline", "(b) Ayahuasca vs. Placebo: Healthy Controls", "(c) Ayahuasca vs. Placebo: TRD Patients"))
data$condition <- factor(data$condition, levels = c("All variables", "Selected variables"))

# Set the order of metrics
data$metric <- factor(data$metric, levels = c("Accuracy", "Sensitivity", "Specificity", "AUROC"))

# Set the colors for each condition
colors <- c("#004e66B2", "#d6a600B2")

ggplot(data, aes(x = value, y = metric, fill = condition)) +
  geom_col(position = "dodge", width = 0.8) +
  facet_wrap(~ group, ncol = 1, scales = "free_y") +
  scale_fill_manual(values = colors) +
  labs(x = "Metric Value", y = "") +
  theme_bw() +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(size = 0.2),
    axis.ticks = element_line(size = 0.2),
    axis.text = element_text(size = 8),
    axis.title = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_blank(),
    legend.text = element_text(size = 8),
    plot.margin = unit(c(1, 1, 0.5, 0.5), "lines")
  ) +
  coord_cartesian(xlim = c(0.4, 0.90)) +
  scale_x_continuous(breaks = seq(0.4, 0.90, by = 0.05))

ggsave("metrics_comparison.png", width = 6, height = 4, dpi = 300)
