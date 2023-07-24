library(ggpubr)
library(tidyverse)
# Selected Features 1

# C-Reactive Protein
plot1fig1 <- ggboxplot(h1dt, x = "group", y = "pcr",
                      color = "group", palette = "grey",
                      add = "jitter", ylab = "C-Reactive Protein (mg/dL)", xlab = "",
                      legend = "none") +
  stat_compare_means(method = "wilcox.test", label = "p.signif", size = 5) +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) +
  theme(legend.position = "right")

# Total Cholesterol
plot2fig1 <- ggboxplot(h1dt, x = "group", y = "totalcholesterol",
                      color = "group", palette = "grey",
                      add = "jitter", ylab = "Total Cholesterol (mg/dL)", xlab = "",
                      legend = "none") +
  stat_compare_means(method = "wilcox.test", label = "p.signif", size = 5) +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) +
  theme(legend.position = "right")



# Awakening Salivary Cortisol
plot3fig1 <- ggboxplot(h1dt, x = "group", y = "aucsalivar",
                      color = "group", palette = "grey",
                      add = "jitter", ylab = "Awakening Salivary Cortisol (nmol/L)", xlab = "",
                      legend = "none") +
  stat_compare_means(method = "wilcox.test", label = "p.signif", size = 5) +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) +
  theme(legend.position = "right")

# Arrange plots
fig <- ggarrange(plot1fig1, plot2fig1, plot3fig1, ncol = 3, nrow = 1, 
                 common.legend = TRUE, legend = "right")

# Remove p-value text from the plot
fig <- fig + theme(legend.text = element_blank())
fig


 #save in a good resolution
ggsave("selectedfeatures1.png", fig, dpi = 300, width = 8, height = 4)

#Selected features 2

# C-Reactive Protein
plot1 <- ggboxplot(h4dt, x = "treatment", y = "pcr",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "C-Reactive Protein (mg/dL)", xlab = "",
                   legend = "none") +
  stat_compare_means(method = "wilcox.test", label = "p.signif", size = 7) +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) +
  theme(legend.position = "none", axis.text.x = element_text(size = 10),
        axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# Glucose
plot2 <- ggboxplot(h4dt, x = "treatment", y = "glucose",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Glucose (mg/dL)", xlab = "",
                   legend = "none") +
  stat_compare_means(method = "wilcox.test", label = "p.signif", size = 7) +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# Lymphocyte
plot3 <- ggboxplot(h4dt, x = "treatment", y = "lymphocyte",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Lymphocyte (%)", xlab = "",
                   legend = "none") +
  stat_compare_means(method = "wilcox.test", label = "p.signif", size = 7) +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

 # Aspartato Aminotransferase (TGO)
plot4 <- ggboxplot(h4dt, x = "treatment", y = "tgo",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Aspartato Aminotransferase (AST) (U/L)", xlab = "",
                   legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))


# TGP
plot5 <- ggboxplot(h4dt, x = "treatment", y = "tgp",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Alanine Aminotransferase (ALT) (U/L)", xlab = "",
                   legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# Hemoglobin
plot6 <- ggboxplot(h4dt, x = "treatment", y = "hemoglobin",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Hemoglobin (g/dL)", xlab = "",
                   legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# RBC
plot7 <- ggboxplot(h4dt, x = "treatment", y = "rbc",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Red Blood Cells (10^6/µL)", xlab = "",
                   legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# LDL Cholesterol
plot8 <- ggboxplot(h4dt, x = "treatment", y = "ldlcholesterol",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "LDL Cholesterol (mg/dL)", xlab = "",
                   legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))
# Segmented
plot9 <- ggboxplot(h4dt, x = "treatment", y = "segmented",
                   color = "treatment", palette = "grey",
                   add = "jitter", ylab = "Segmented Neutrophils (10^3/µL)", xlab = "",
                   legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))
# Platelets
plot10 <- ggboxplot(h4dt, x = "treatment", y = "platelets",
                    color = "treatment", palette = "grey",
                    add = "jitter", ylab = "Platelets (x10^3/uL)", xlab = "",
                    legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# BDNF
plot11 <- ggboxplot(h4dt, x = "treatment", y = "bdnf",
                    color = "treatment", palette = "grey",
                    add = "jitter", ylab = "BDNF (ng/mL)", xlab = "",
                    legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# Triglycerides
plot12 <- ggboxplot(h4dt, x = "treatment", y = "triglycerides",
                    color = "treatment", palette = "grey",
                    add = "jitter", ylab = "Triglycerides (mg/dL)", xlab = "",
                    legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# Leucocytes
plot13 <- ggboxplot(h4dt, x = "treatment", y = "leukocytes",
                    color = "treatment", palette = "grey",
                    add = "jitter", ylab = "Leucocytes (x10^3/uL)", xlab = "",
                    legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))

# Hematocrit
plot14 <- ggboxplot(h4dt, x = "treatment", y = "hematocrit",
                    color = "treatment", palette = "grey",
                    add = "jitter", ylab = "Hematocrit (%)", xlab = "",
                    legend = "none") +
  theme(plot.title = element_blank()) +
  scale_colour_grey(start = 0.2, end = 0.6) + theme(legend.position = "none", axis.text.x = element_text(size = 10),
                                                    axis.text.y = element_text(size = 10)) +
  scale_x_discrete(labels = c("Placebo", "Ayahuasca"))
# Adjust the size of the plots
plot_width <- 5  # Modify as needed
plot_height <- 4  # Modify as needed

# Arrange the plots into three categories
hematological_plots <- ggarrange(plot6, plot7, plot10, plot14, ncol = 2, nrow = 2, labels = c("a", "b", "c", "d"), widths = rep(plot_width, 2), heights = rep(plot_height, 2))
inflammatory_plots <- ggarrange(plot1, plot3, plot4, plot5, plot13, plot9, plot11, ncol = 4, nrow = 2, labels = c("a", "b", "c", "d", "e", "f", "g"), widths = rep(plot_width, 3), heights = rep(plot_height, 3))
metabolic_plots <- ggarrange(plot2, plot8, plot12, ncol = 3, nrow = 1, labels = c("a", "b", "c"), widths = rep(plot_width, 3), heights = rep(plot_height, 1))

# Save the plots in high quality with appropriate names
ggsave("hematological.png", hematological_plots, dpi = 300, width = 10, height = 8)  # Adjust width and height as needed
ggsave("inflammatory.png", inflammatory_plots, dpi = 300, width = 14, height = 10)  # Adjust width and height as needed
ggsave("metabolic.png", metabolic_plots, dpi = 300, width = 12, height = 6)  # Adjust width and height as needed

