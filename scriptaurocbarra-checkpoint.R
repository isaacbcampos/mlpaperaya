library(tidyverse)

## Figure AUROC

data <- tribble(
  ~metric, ~value,        ~group,
  "Baseline All variables", 0.60, "Controls x Patients", 
  "Baseline Selected variables", 0.65, "Controls x Patients", 
  "Controls All variables",   0.56, "Ayahuasca x Placebo",
  "Controls Selected Variables",   0.53, "Ayahuasca x Placebo",
  "TDR Patients All variables",   0.71, "Ayahuasca x Placebo",
  "TDR Patients Selected variables",   0.80, "Ayahuasca x Placebo"
)

# Set the order of the groups
data$group <- factor(data$group, levels = c("Controls x Patients", "Ayahuasca x Placebo"))

# Set the colors for each group
colors <- c("#004e66B2", "#d6a600B2")

ggplot(data, aes(x = value, y = metric, color = group)) +
  geom_segment(aes(xend = 0.5, yend = metric), size = 5, color = "gray") +
  geom_segment(aes(x = 0.5, xend = value, y = metric, yend = metric, color = group), size = 5, alpha = 0.8) +
  scale_color_manual(values = colors) +
  facet_wrap(~ group, ncol = 1, scales = "free_y") +
  labs(x = "Mean AUROC", y = "") +
  theme_bw() +
  theme(
    panel.grid.major = element_line(size = 0.2),
    panel.grid.minor = element_blank(),
    panel.border = element_blank(),
    axis.line = element_line(size = 0.2),
    axis.ticks = element_line(size = 0.2),
    axis.text = element_text(size = 10),
    axis.title = element_text(size = 10),
    legend.position = "none",
    plot.margin = unit(c(1, 1, 0.5, 0.5), "lines")
  ) +
  scale_x_continuous(limits = c(0.50, 0.85), breaks = seq(0.50, 0.85, by = 0.05))
ggsave("aurocbars.png", width = 6, height = 4, dpi = 300)
