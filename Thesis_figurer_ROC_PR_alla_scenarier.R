# ============================================================
# Thesis_figurer_ROC_PR_alla_scenarier.R
#
# Genererar ROC- och PR-kurvor för ALLA tre scenarier:
#   Scenario 1 – Utan balanseringsåtgärd      → Figur 4.1 & 4.2
#   Scenario 2 – SMOTE         → Figur 4.3 & 4.4
#   Scenario 3 – Class Weights → Figur 4.5 & 4.6
#
# Förutsätter att alla tre ML-skript körts och att dessa
# objekt finns i Environment:
#   pred_all_no_smote
#   pred_all_smote
#   pred_all_cw
# ============================================================

library(tidymodels)
library(ggplot2)
library(dplyr)

# ── Kontrollera att alla objekt finns ──────────────────────
for (obj in c("pred_all_no_smote", "pred_all_smote", "pred_all_cw")) {
  if (!exists(obj)) stop(paste("Saknas:", obj, "– kör ML-skripten först."))
}

# ── Gemensamma inställningar ───────────────────────────────
model_colors <- c(
  "Logistic Regression"           = "#D7191C",
  "Weighted Logistic Regression"  = "#D7191C",
  "LASSO Logistic Regression"     = "#F59E0B",
  "Weighted LASSO Logistic Regression" = "#F59E0B",
  "Random Forest"                 = "#1A9641",
  "Weighted Random Forest"        = "#1A9641",
  "XGBoost"                       = "#2C7BB6",
  "Weighted XGBoost"              = "#2C7BB6"
)

base_theme <- theme_classic(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(color = "grey40", size = 10,
                                    margin = margin(b = 10)),
    legend.position  = "bottom",
    legend.text      = element_text(size = 9),
    legend.key.width = unit(1.6, "cm"),
    legend.spacing.y = unit(0.3, "cm"),
    axis.title       = element_text(size = 11),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.4),
    plot.margin      = margin(15, 20, 10, 15)
  )

# ── Hjälpfunktion: gör ROC-plot ────────────────────────────
make_roc_plot <- function(pred_data, fig_nr, scenario_label) {

  # AUC per modell
  auc_vals <- pred_data %>%
    group_by(model) %>%
    roc_auc(truth = Biopsy, .pred_Yes, event_level = "second") %>%
    mutate(label = paste0(model, "  (AUC = ", round(.estimate, 3), ")"))

  # Lägg label på pred-data
  pred_labeled <- pred_data %>%
    left_join(auc_vals %>% select(model, label), by = "model") %>%
    mutate(label = factor(label))

  # Färgmappning med labels
  col_map <- setNames(
    model_colors[auc_vals$model],
    auc_vals$label
  )

  # Kurv-data
  curve_data <- pred_labeled %>%
    group_by(label) %>%
    roc_curve(truth = Biopsy, .pred_Yes, event_level = "second")

  ggplot(curve_data,
         aes(x = 1 - specificity, y = sensitivity, color = label)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "grey55", linewidth = 0.8) +
    geom_path(linewidth = 1.1, alpha = 0.95) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_color_manual(values = col_map) +
    annotate("text", x = 0.62, y = 0.10,
             label = "Slumpmässig klassificerare",
             color = "grey50", size = 3, angle = 36, fontface = "italic") +
    labs(
      title    = paste0("ROC-kurvor – ", scenario_label),
      subtitle = "Streckad diagonal = slumpmässig klassificerare (AUC = 0.50)",
      x        = "1 − Specificitet  (False Positive Rate)",
      y        = "Sensitivitet  (True Positive Rate)",
      color    = NULL
    ) +
    base_theme +
    guides(color = guide_legend(nrow = 2, byrow = TRUE))
}

# ── Hjälpfunktion: gör PR-plot ─────────────────────────────
make_pr_plot <- function(pred_data, fig_nr, scenario_label) {

  prevalens <- mean(pred_data$Biopsy == "Yes")

  auc_vals <- pred_data %>%
    group_by(model) %>%
    pr_auc(truth = Biopsy, .pred_Yes, event_level = "second") %>%
    mutate(label = paste0(model, "  (AUC = ", round(.estimate, 3), ")"))

  pred_labeled <- pred_data %>%
    left_join(auc_vals %>% select(model, label), by = "model") %>%
    mutate(label = factor(label))

  col_map <- setNames(
    model_colors[auc_vals$model],
    auc_vals$label
  )

  curve_data <- pred_labeled %>%
    group_by(label) %>%
    pr_curve(truth = Biopsy, .pred_Yes, event_level = "second")

  ggplot(curve_data,
         aes(x = recall, y = precision, color = label)) +
    geom_hline(yintercept = prevalens,
               linetype = "dashed", color = "grey55", linewidth = 0.8) +
    geom_path(linewidth = 1.1, alpha = 0.95) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_color_manual(values = col_map) +
    annotate("text",
             x = 0.60, y = prevalens + 0.04,
             label = paste0("Baslinje (prevalens = ", round(prevalens, 3), ")"),
             color = "grey45", size = 3, fontface = "italic") +
    labs(
      title    = paste0("Precision-Recall-kurvor – ", scenario_label),
      subtitle = paste0("Streckad linje = naiv baslinje (prevalens ≈ ",
                        round(prevalens, 3), ")"),
      x        = "Recall  (Sensitivitet)",
      y        = "Precision",
      color    = NULL
    ) +
    base_theme +
    guides(color = guide_legend(nrow = 2, byrow = TRUE))
}

# ── Spara hjälpfunktion ────────────────────────────────────
save_fig <- function(plot, filnamn) {
  ggsave(filnamn, plot = plot, width = 7, height = 6, dpi = 300, bg = "white")
  cat("Sparad:", filnamn, "\n")
}

# ============================================================
# SCENARIO 1 – UTAN BALANSERINGSÅTGÄRD
# ============================================================
cat("\n--- Scenario 1: Utan balanseringsåtgärd ---\n")

fig_4_1 <- make_roc_plot(pred_all_no_smote,
                          fig_nr = "4.1",
                          scenario_label = "Scenario 1: Utan balanseringsåtgärd")
print(fig_4_1)
save_fig(fig_4_1, "ROC_utan_balansering.png")

fig_4_2 <- make_pr_plot(pred_all_no_smote,
                         fig_nr = "4.2",
                         scenario_label = "Scenario 1: Utan balanseringsåtgärd")
print(fig_4_2)
save_fig(fig_4_2, "PR_utan_balansering.png")

# ============================================================
# SCENARIO 2 – SMOTE
# ============================================================
cat("\n--- Scenario 2: SMOTE ---\n")

fig_4_3 <- make_roc_plot(pred_all_smote,
                          fig_nr = "4.3",
                          scenario_label = "Scenario 2: SMOTE")
print(fig_4_3)
save_fig(fig_4_3, "ROC_SMOTE.png")

fig_4_4 <- make_pr_plot(pred_all_smote,
                         fig_nr = "4.4",
                         scenario_label = "Scenario 2: SMOTE")
print(fig_4_4)
save_fig(fig_4_4, "PR_SMOTE.png")

# ============================================================
# SCENARIO 3 – CLASS WEIGHTS
# ============================================================
cat("\n--- Scenario 3: Class Weights ---\n")

fig_4_5 <- make_roc_plot(pred_all_cw,
                          fig_nr = "4.5",
                          scenario_label = "Scenario 3: Class Weights")
print(fig_4_5)
save_fig(fig_4_5, "ROC_class_weights.png")

fig_4_6 <- make_pr_plot(pred_all_cw,
                         fig_nr = "4.6",
                         scenario_label = "Scenario 3: Class Weights")
print(fig_4_6)
save_fig(fig_4_6, "PR_class_weights.png")

# ============================================================
# BONUS: Kombinerad jämförelse – Random Forest i alla scenarier
# Figur 4.7 (ROC) och Figur 4.8 (PR) – bra för avsnitt 4.4
# ============================================================
cat("\n--- Bonus: RF-jämförelse alla scenarier ---\n")

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)

  # Slå ihop RF från alla tre scenarier
  rf_combined <- bind_rows(
    pred_all_no_smote %>%
      filter(model == "Random Forest") %>%
      mutate(scenario = "Scenario 1: Utan balanseringsåtgärd"),
    pred_all_smote %>%
      filter(model == "Random Forest") %>%
      mutate(scenario = "Scenario 2: SMOTE"),
    pred_all_cw %>%
      filter(model == "Weighted Random Forest") %>%
      mutate(scenario = "Scenario 3: Class Weights")
  )

  scenario_colors <- c(
    "Scenario 1: Utan balanseringsåtgärd"       = "#D7191C",
    "Scenario 2: SMOTE"          = "#1A9641",
    "Scenario 3: Class Weights"  = "#2C7BB6"
  )

  # AUC-labels per scenario
  roc_sc_auc <- rf_combined %>%
    group_by(scenario) %>%
    roc_auc(truth = Biopsy, .pred_Yes, event_level = "second") %>%
    mutate(label = paste0(scenario, "  (AUC = ", round(.estimate, 3), ")"))

  pr_sc_auc <- rf_combined %>%
    group_by(scenario) %>%
    pr_auc(truth = Biopsy, .pred_Yes, event_level = "second") %>%
    mutate(label = paste0(scenario, "  (AUC = ", round(.estimate, 3), ")"))

  rf_roc_labeled <- rf_combined %>%
    left_join(roc_sc_auc %>% select(scenario, label), by = "scenario")

  rf_pr_labeled <- rf_combined %>%
    left_join(pr_sc_auc %>% select(scenario, label), by = "scenario")

  roc_sc_colors <- setNames(
    scenario_colors[roc_sc_auc$scenario], roc_sc_auc$label)
  pr_sc_colors  <- setNames(
    scenario_colors[pr_sc_auc$scenario],  pr_sc_auc$label)

  roc_rf_data <- rf_roc_labeled %>%
    group_by(label) %>%
    roc_curve(truth = Biopsy, .pred_Yes, event_level = "second")

  pr_rf_data <- rf_pr_labeled %>%
    group_by(label) %>%
    pr_curve(truth = Biopsy, .pred_Yes, event_level = "second")

  prevalens <- mean(rf_combined$Biopsy == "Yes")

  fig_4_7 <- ggplot(roc_rf_data,
                    aes(x = 1 - specificity, y = sensitivity, color = label)) +
    geom_abline(slope = 1, intercept = 0,
                linetype = "dashed", color = "grey55", linewidth = 0.8) +
    geom_path(linewidth = 1.3) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_color_manual(values = roc_sc_colors) +
    labs(title    = "ROC-kurvor – Random Forest per scenario",
         subtitle = "Jämförelse av Utan balanseringsåtgärd vs SMOTE vs Class Weights",
         x = "1 − Specificitet", y = "Sensitivitet", color = NULL) +
    base_theme +
    guides(color = guide_legend(nrow = 3))

  fig_4_8 <- ggplot(pr_rf_data,
                    aes(x = recall, y = precision, color = label)) +
    geom_hline(yintercept = prevalens,
               linetype = "dashed", color = "grey55", linewidth = 0.8) +
    geom_path(linewidth = 1.3) +
    scale_x_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_y_continuous(limits = c(0,1), breaks = seq(0,1,0.2)) +
    scale_color_manual(values = pr_sc_colors) +
    labs(title    = "PR-kurvor – Random Forest per scenario",
         subtitle = "Jämförelse av Utan balanseringsåtgärd vs SMOTE vs Class Weights",
         x = "Recall", y = "Precision", color = NULL) +
    base_theme +
    guides(color = guide_legend(nrow = 3))

  print(fig_4_7); save_fig(fig_4_7, "ROC_RF_scenarion.png")
  print(fig_4_8); save_fig(fig_4_8, "PR_RF_scenarion.png")

  # Kombinerad
  fig_combined <- (fig_4_7 | fig_4_8) +
    plot_annotation(
      title = "Random Forest: ROC och PR per scenario",
      theme = theme(plot.title = element_text(face = "bold", size = 14))
    )
  save_fig(fig_combined %>% { ggsave("Figur_kombinerad.png",
    plot = fig_combined, width = 14, height = 6, dpi = 300, bg = "white"); . },
    "Figur_kombinerad.png")
}

cat("\n=== Alla figurer klara! ===\n\n")
cat("Scenario 1 (Utan balanseringsåtgärd):\n")
cat("  ROC_utan_balansering.png\n")
cat("  PR_utan_balansering.png\n\n")
cat("Scenario 2 (SMOTE):\n")
cat("  ROC_SMOTE.png\n")
cat("  PR_SMOTE.png\n\n")
cat("Scenario 3 (Class Weights):\n")
cat("  ROC_class_weights.png\n")
cat("  PR_class_weights.png\n\n")
cat("Jämförelse RF alla scenarier:\n")
cat("  ROC_RF_scenarion.png\n")
cat("  PR_RF_scenarion.png\n")
