# ============================================================
# Thesis_hantera_NA.R  –  Datahantering & EDA
#
# FÖRBÄTTRINGAR vs originalversionen:
#   1) Relativ sökväg (reproducerbar på alla datorer)
#   2) Imputation sker INTE här – den görs inne i recipe
#      för att undvika data leakage från testdata
#   3) NZV-kontroll sker som information, inte som borttagning
#      (step_zv i recipe hanterar det korrekt per fold)
#   4) Explorativ dataanalys (EDA) tillagd:
#      - Missingness-visualisering
#      - Boxplottar för kontinuerliga variabler
#      - Stapeldiagram för binära variabler
#      - Åldersfördelning
#      - Spearmans korrelationsmatris
# ============================================================

library(tidyverse)
library(naniar)
library(corrplot)
library(janitor)

# ------------------------------------------------------------
# 0) Läs in rådata med relativ sökväg
#    Lägg CSV-filen i samma mapp som detta skript, eller
#    justera sökvägen relativt till ditt projekt.
# ------------------------------------------------------------
df_raw <- read.csv(
  file.path("Dataset", "risk_factors_cervical_cancer.csv"),
  na.strings = c("?", "", "NA"),
  check.names = FALSE
)

cat("Start: rader =", nrow(df_raw), "kolumner =", ncol(df_raw), "\n")

# ------------------------------------------------------------
# 1) Ta bort variabler utan data (noll variation i hela kolumnen)
# ------------------------------------------------------------
drop_no_data <- c("STDs:AIDS", "STDs:cervical condylomatosis")
drop_no_data <- intersect(drop_no_data, names(df_raw))
if (length(drop_no_data) > 0) {
  df_raw <- df_raw[, setdiff(names(df_raw), drop_no_data), drop = FALSE]
  cat("Tog bort (ingen data):", paste(drop_no_data, collapse = ", "), "\n")
}

# ------------------------------------------------------------
# 2) Ta bort dataleakage-variabler och övriga målvariabler
#    Dessa är resultat av samma undersökning som Biopsy –
#    att inkludera dem ger artificiellt bra modeller.
# ------------------------------------------------------------
leakage_vars <- c("Dx", "Dx:Cancer", "Dx:CIN", "Dx:HPV",
                  "Hinselmann", "Schiller", "Citology")
df <- df_raw %>% select(-any_of(leakage_vars))
cat("Tog bort leakage-variabler:", paste(intersect(leakage_vars, names(df_raw)), collapse = ", "), "\n")

# ------------------------------------------------------------
# 3) Ta bort variabler med extrem missingness (>90% NA)
# ------------------------------------------------------------
drop_high_na <- c("STDs: Time since last diagnosis",
                  "STDs: Time since first diagnosis")
drop_high_na <- intersect(drop_high_na, names(df))
if (length(drop_high_na) > 0) {
  df <- df[, setdiff(names(df), drop_high_na), drop = FALSE]
  cat("Tog bort (>90% NA):", paste(drop_high_na, collapse = ", "), "\n")
}

cat("Efter kolumnborttagning: rader =", nrow(df), "kolumner =", ncol(df), "\n")

# ------------------------------------------------------------
# 4) Visa missingness-sammanfattning
# ------------------------------------------------------------
miss_summary <- df %>%
  summarise(across(everything(), ~ sum(is.na(.)))) %>%
  pivot_longer(everything(),
               names_to  = "variabel",
               values_to = "n_saknade") %>%
  mutate(pct_saknade = round(n_saknade / nrow(df) * 100, 1)) %>%
  arrange(desc(n_saknade)) %>%
  filter(n_saknade > 0)

cat("\nMissingness per variabel:\n")
print(miss_summary, n = 40)

# Visualisera missingness
gg_miss_var(df, show_pct = TRUE) +
  labs(title = "Andel saknade värden per variabel (%)",
       x = "% saknade", y = "") +
  theme_minimal()

# ------------------------------------------------------------
# 5) Little's MCAR-test
#    H0: data saknas helt slumpmässigt (MCAR)
#    p < 0.05 -> data är INTE MCAR -> motiverar imputation
# ------------------------------------------------------------
mcar_result <- mcar_test(df)
cat("\nLittle's MCAR-test:\n")
print(mcar_result)

# ------------------------------------------------------------
# 6) Koda om Biopsy för EDA-visualisering
# ------------------------------------------------------------
df_eda <- df %>%
  mutate(
    Biopsy = case_when(
      Biopsy %in% c(1, "1") ~ "Positiv (Yes)",
      Biopsy %in% c(0, "0") ~ "Negativ (No)",
      TRUE ~ as.character(Biopsy)
    ),
    Biopsy = factor(Biopsy, levels = c("Negativ (No)", "Positiv (Yes)"))
  ) %>%
  filter(!is.na(Biopsy))

cat("\nKlassfördelning Biopsy:\n")
print(table(df_eda$Biopsy))
print(prop.table(table(df_eda$Biopsy)))

# ------------------------------------------------------------
# 7) EDA – Boxplottar för kontinuerliga variabler
# ------------------------------------------------------------
continuous_vars <- c("Age", "Number of sexual partners",
                     "First sexual intercourse", "Num of pregnancies",
                     "Hormonal Contraceptives (years)", "IUD (years)",
                     "STDs (number)")
continuous_vars <- intersect(continuous_vars, names(df_eda))

df_eda %>%
  select(Biopsy, all_of(continuous_vars)) %>%
  mutate(across(-Biopsy, as.numeric)) %>%
  pivot_longer(-Biopsy) %>%
  ggplot(aes(x = Biopsy, y = value, fill = Biopsy)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  facet_wrap(~name, scales = "free_y") +
  scale_fill_manual(values = c("Negativ (No)" = "#4DBBAE",
                               "Positiv (Yes)" = "#E05A5A")) +
  labs(title = "Kontinuerliga variabler per biopsiutfall",
       x = NULL, y = NULL, fill = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom",
        strip.text = element_text(size = 8))

# ------------------------------------------------------------
# 8) EDA – Andel positiva för binära variabler
# ------------------------------------------------------------
binary_vars <- c("Smokes", "Hormonal Contraceptives", "IUD", "STDs",
                 "STDs:condylomatosis", "STDs:vulvo-perineal condylomatosis",
                 "STDs:syphilis", "STDs:HIV", "STDs:HPV")
binary_vars <- intersect(binary_vars, names(df_eda))

if (length(binary_vars) > 0) {
  df_eda %>%
    select(Biopsy, all_of(binary_vars)) %>%
    mutate(across(-Biopsy, as.numeric)) %>%
    pivot_longer(-Biopsy) %>%
    group_by(Biopsy, name) %>%
    summarise(andel = mean(value, na.rm = TRUE), .groups = "drop") %>%
    ggplot(aes(x = reorder(name, andel), y = andel, fill = Biopsy)) +
    geom_col(position = "dodge", alpha = 0.85) +
    coord_flip() +
    scale_fill_manual(values = c("Negativ (No)" = "#4DBBAE",
                                 "Positiv (Yes)" = "#E05A5A")) +
    labs(title = "Andel positiva (binära variabler) per biopsiutfall",
         x = NULL, y = "Andel", fill = NULL) +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# ------------------------------------------------------------
# 9) EDA – Åldersfördelning
# ------------------------------------------------------------
df_eda %>%
  mutate(Age = as.numeric(Age)) %>%
  ggplot(aes(x = Age, fill = Biopsy)) +
  geom_histogram(bins = 30, position = "identity", alpha = 0.6) +
  scale_fill_manual(values = c("Negativ (No)" = "#4DBBAE",
                               "Positiv (Yes)" = "#E05A5A")) +
  labs(title = "Åldersfördelning per biopsiutfall",
       x = "Ålder", y = "Antal", fill = NULL) +
  theme_minimal()

# ------------------------------------------------------------
# 10) EDA – Spearmans korrelationsmatris (prediktorer)
# ------------------------------------------------------------
cor_data <- df %>%
  select(-any_of(c("Biopsy"))) %>%
  mutate(across(everything(), ~ suppressWarnings(as.numeric(as.character(.x))))) %>%
  select(where(~ sum(!is.na(.x)) > 10))  # ta bara med kolumner med tillräckligt data

cor_mat <- cor(cor_data, method = "spearman", use = "pairwise.complete.obs")

corrplot(cor_mat,
         method = "color", type = "upper", order = "hclust",
         tl.cex = 0.6, tl.col = "black",
         title  = "Spearmans korrelationsmatris – prediktorer",
         mar    = c(0, 0, 2, 0))

# ------------------------------------------------------------
# 11) VIKTIGT: Spara df UTAN imputation
#     Imputation sker inne i recipe i ML-skripten för att
#     undvika data leakage (testdata-medianer ska inte
#     påverka träningsimputationen).
# ------------------------------------------------------------
cat("\nSlut preprocessing: rader =", nrow(df), "kolumner =", ncol(df), "\n")
cat("NA kvar totalt:", sum(is.na(df)), "\n")
cat("OBS: Imputation sker inuti recipe i ML-skripten (ej här).\n")

# df-objektet är nu klart att användas av ML-skripten.
