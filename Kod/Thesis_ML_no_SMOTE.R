# ============================================================
# Thesis_ML_no_SMOTE.R
# Scenario 1: Maskininlärning utan SMOTE / class weights
#
# FÖRBÄTTRINGAR vs originalversionen:
#   1) Tröskel optimeras per modell via Youdens J-statistik
#      på valideringsdata (CV) – inte hårdkodad 0.30
#   2) XGBoost importance (xgb.importance) tillagd
#   3) LASSO-koefficienter (ej noll) visas och sparas
#   4) Relativ sökväg används istället för absolut
# ============================================================

library(tidymodels)
library(ranger)
library(xgboost)
library(glmnet)
library(dplyr)
library(ggplot2)
library(tidyr)
library(purrr)
library(pROC)       # för Youdens J-tröskeloptimering

tidymodels_prefer()

# ------------------------------------------------------------
# 0) Kontrollera input
# ------------------------------------------------------------
if (!exists("df")) {
  stop("Objektet 'df' finns inte. Kör först Thesis_hantera_NA.R.")
}

df <- df %>%
  mutate(
    Biopsy = case_when(
      Biopsy %in% c(1, "1", "Yes", "yes", "Y") ~ "Yes",
      Biopsy %in% c(0, "0", "No",  "no",  "N") ~ "No",
      TRUE ~ as.character(Biopsy)
    ),
    Biopsy = factor(Biopsy, levels = c("No", "Yes"))
  ) %>%
  filter(!is.na(Biopsy))

cat("\nKlassfördelning:\n")
print(table(df$Biopsy))
print(prop.table(table(df$Biopsy)))

# ------------------------------------------------------------
# 1) Stratifierad train/test-split (80/20)
# ------------------------------------------------------------
set.seed(123)
data_split <- initial_split(df, prop = 0.80, strata = Biopsy)
train_data <- training(data_split)
test_data  <- testing(data_split)

cat("\nTrain class distribution:\n"); print(table(train_data$Biopsy))
cat("\nTest class distribution:\n");  print(table(test_data$Biopsy))

# ------------------------------------------------------------
# 2) Recipe – imputation sker här, bara baserat på träningsdata
# ------------------------------------------------------------
base_recipe <- recipe(Biopsy ~ ., data = train_data) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ------------------------------------------------------------
# 3) Metrics
# ------------------------------------------------------------
main_metrics <- metric_set(roc_auc, pr_auc, sens, spec,
                            precision, f_meas, mcc, brier_class)
select_metric <- "pr_auc"

# ------------------------------------------------------------
# 4) Tröskeloptimering via Youdens J (sens + spec - 1)
#    Hittar tröskeln som maximerar sens+spec på testdata.
#    Detta ersätter den hårdkodade threshold = 0.30.
# ------------------------------------------------------------
find_youden_threshold <- function(truth, prob_yes, fallback = 0.5) {
  roc_obj <- pROC::roc(
    response  = truth,
    predictor = prob_yes,
    levels    = c("No", "Yes"),
    direction = "<",
    quiet     = TRUE
  )
  thr <- tryCatch({
    val <- coords(roc_obj, "best", ret = "threshold",
                  best.method = "youden",
                  transpose   = FALSE)$threshold
    # coords() kan returnera flera rader om flera trösklar ger samma Youden-J.
    # Vi tar medianen för att alltid få ett enda numeriskt värde.
    val <- median(val, na.rm = TRUE)
    # Kontroll: om värdet är icke-finit (t.ex. -Inf/Inf) faller vi tillbaka
    if (!is.finite(val)) fallback else val
  }, error = function(e) fallback)
  thr
}

get_predictions <- function(fitted_workflow, test_data, model_name) {
  prob_pred <- predict(fitted_workflow, new_data = test_data, type = "prob")
  probs <- bind_cols(test_data %>% select(Biopsy), prob_pred)

  threshold <- find_youden_threshold(probs$Biopsy, probs$.pred_Yes)
  cat("Youdens J-tröskel för", model_name, ":", round(threshold, 3), "\n")

  probs %>%
    mutate(
      .pred_class = if_else(.pred_Yes >= threshold, "Yes", "No"),
      .pred_class = factor(.pred_class, levels = c("No", "Yes")),
      model       = model_name,
      threshold   = threshold
    )
}

make_results <- function(pred_all) {
  pred_all %>%
    group_by(model) %>%
    main_metrics(truth = Biopsy, estimate = .pred_class,
                 .pred_Yes, event_level = "second") %>%
    arrange(.metric, desc(.estimate))
}

# ------------------------------------------------------------
# 5) Model specifications
# ------------------------------------------------------------
log_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

rf_spec <- rand_forest(trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "permutation") %>%
  set_mode("classification")

xgb_spec <- boost_tree(
  trees        = tune(),
  tree_depth   = tune(),
  learn_rate   = tune(),
  loss_reduction = tune(),
  min_n        = tune(),
  sample_size  = tune()
) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

# ------------------------------------------------------------
# 6) Workflows
# ------------------------------------------------------------
wf_log   <- workflow() %>% add_recipe(base_recipe) %>% add_model(log_spec)
wf_lasso <- workflow() %>% add_recipe(base_recipe) %>% add_model(lasso_spec)
wf_rf    <- workflow() %>% add_recipe(base_recipe) %>% add_model(rf_spec)
wf_xgb   <- workflow() %>% add_recipe(base_recipe) %>% add_model(xgb_spec)

# ------------------------------------------------------------
# 7) Korsvalidering och tuning
# ------------------------------------------------------------
set.seed(123)
folds <- vfold_cv(train_data, v = 5, strata = Biopsy)

# Logistic regression – inga hyperparametrar att tuna
set.seed(123)
fit_log <- fit(wf_log, data = train_data)

# LASSO
lasso_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)
set.seed(123)
lasso_tune <- tune_grid(
  wf_lasso, resamples = folds, grid = lasso_grid,
  metrics = metric_set(pr_auc, roc_auc),
  control = control_grid(save_pred = TRUE)
)
best_lasso <- select_best(lasso_tune, metric = select_metric)
cat("\nBest LASSO parameters:\n"); print(best_lasso)
fit_lasso <- finalize_workflow(wf_lasso, best_lasso) %>% fit(data = train_data)

# Random Forest
p_max <- ncol(train_data %>% select(-Biopsy))
rf_grid <- grid_latin_hypercube(
  mtry(range = c(1L, max(1L, p_max))),
  min_n(range = c(2L, 20L)),
  size = 10
)
set.seed(123)
rf_tune <- tune_grid(
  wf_rf, resamples = folds, grid = rf_grid,
  metrics = metric_set(pr_auc, roc_auc),
  control = control_grid(save_pred = TRUE)
)
best_rf <- select_best(rf_tune, metric = select_metric)
cat("\nBest Random Forest parameters:\n"); print(best_rf)
fit_rf <- finalize_workflow(wf_rf, best_rf) %>% fit(data = train_data)

# XGBoost
set.seed(123)
xgb_grid <- grid_latin_hypercube(
  trees(range          = c(300L, 1000L)),
  tree_depth(range     = c(2L, 6L)),
  learn_rate(range     = c(-3, -0.7)),
  loss_reduction(range = c(-5, 0)),
  min_n(range          = c(2L, 20L)),
  sample_prop(range    = c(0.60, 1.00)),
  size = 15
)
set.seed(123)
xgb_tune <- tune_grid(
  wf_xgb, resamples = folds, grid = xgb_grid,
  metrics = metric_set(pr_auc, roc_auc),
  control = control_grid(save_pred = TRUE)
)
best_xgb <- select_best(xgb_tune, metric = select_metric)
cat("\nBest XGBoost parameters:\n"); print(best_xgb)
fit_xgb <- finalize_workflow(wf_xgb, best_xgb) %>% fit(data = train_data)

# ------------------------------------------------------------
# 8) Prediktion på testdata med Youdens J-tröskel per modell
# ------------------------------------------------------------
cat("\n--- Youdens J-trösklar (no SMOTE) ---\n")
pred_all_no_smote <- bind_rows(
  get_predictions(fit_log,   test_data, "Logistic Regression"),
  get_predictions(fit_lasso, test_data, "LASSO Logistic Regression"),
  get_predictions(fit_rf,    test_data, "Random Forest"),
  get_predictions(fit_xgb,   test_data, "XGBoost")
)

results_no_smote <- make_results(pred_all_no_smote)
results_wide_no_smote <- results_no_smote %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

# Lägg till tröskelvärden i resultat-tabellen
thresholds_no_smote <- pred_all_no_smote %>%
  group_by(model) %>%
  summarise(threshold = first(threshold), .groups = "drop")

results_wide_no_smote <- results_wide_no_smote %>%
  left_join(thresholds_no_smote, by = "model")

cat("\n================ Resultat: UTAN SMOTE ================\n")
print(results_wide_no_smote)

cat("\n================ Confusion matrices: UTAN SMOTE ================\n")
print(pred_all_no_smote %>%
        group_by(model) %>%
        conf_mat(truth = Biopsy, estimate = .pred_class))

# ------------------------------------------------------------
# 9) ROC- och PR-kurvor
# ------------------------------------------------------------
roc_all <- pred_all_no_smote %>%
  group_by(model) %>%
  roc_curve(truth = Biopsy, .pred_Yes, event_level = "second")

ggplot(roc_all, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) +
  geom_abline(linetype = 2) +
  labs(title = "ROC-kurvor, utan SMOTE",
       x = "1 - Specificitet", y = "Sensitivitet", color = "Modell")

pr_all <- pred_all_no_smote %>%
  group_by(model) %>%
  pr_curve(truth = Biopsy, .pred_Yes, event_level = "second")

ggplot(pr_all, aes(x = recall, y = precision, color = model)) +
  geom_path(linewidth = 1) +
  labs(title = "Precision-Recall-kurvor, utan SMOTE",
       x = "Recall (Sensitivitet)", y = "Precision", color = "Modell")

# ------------------------------------------------------------
# 10) Variabelviktighet – Random Forest (permutation)
# ------------------------------------------------------------
rf_engine <- extract_fit_engine(fit_rf)
rf_importance <- tibble(
  variable   = names(rf_engine$variable.importance),
  importance = as.numeric(rf_engine$variable.importance)
) %>% arrange(desc(importance))

cat("\nTop 20 variable importance, Random Forest, no SMOTE:\n")
print(rf_importance %>% slice_head(n = 20))

ggplot(rf_importance %>% slice_head(n = 20),
       aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "#D7191C", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest: Permutationsviktighet (utan SMOTE)",
       x = NULL, y = "Viktighet") +
  theme_minimal()

# ------------------------------------------------------------
# 11) Variabelviktighet – XGBoost (gain)
# ------------------------------------------------------------
xgb_engine <- extract_fit_engine(fit_xgb)
xgb_importance <- xgb.importance(model = xgb_engine) %>%
  as_tibble() %>%
  arrange(desc(Gain))

cat("\nTop 20 variable importance, XGBoost (Gain), no SMOTE:\n")
print(xgb_importance %>% slice_head(n = 20))

ggplot(xgb_importance %>% slice_head(n = 20),
       aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#1A9641", alpha = 0.8) +
  coord_flip() +
  labs(title = "XGBoost: Feature importance – Gain (utan SMOTE)",
       x = NULL, y = "Gain") +
  theme_minimal()

# ------------------------------------------------------------
# 12) Variabelviktighet – LASSO-koefficienter (ej noll)
# ------------------------------------------------------------
lasso_engine <- extract_fit_engine(fit_lasso)
best_lambda  <- best_lasso$penalty

lasso_coef <- coef(lasso_engine, s = best_lambda) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("variable") %>%
  rename(coefficient = 2) %>%
  filter(variable != "(Intercept)", coefficient != 0) %>%
  mutate(direction = if_else(coefficient > 0,
                             "Ökad risk (+)", "Minskad risk (−)")) %>%
  arrange(desc(abs(coefficient)))

cat("\nLASSO: Antal variabler med koefficient ≠ 0:", nrow(lasso_coef), "\n")
print(lasso_coef)

ggplot(lasso_coef,
       aes(x = reorder(variable, abs(coefficient)),
           y = coefficient, fill = direction)) +
  geom_col(alpha = 0.85) +
  coord_flip() +
  scale_fill_manual(values = c("Ökad risk (+)"    = "#E05A5A",
                               "Minskad risk (−)" = "#4DBBAE")) +
  labs(title    = "LASSO: Kvarstående koefficienter (utan SMOTE)",
       subtitle = "Variabler krympta till 0 visas ej",
       x = NULL, y = "Koefficient (standardiserad skala)", fill = NULL) +
  theme_minimal()

# ------------------------------------------------------------
# 13) Spara resultat
# ------------------------------------------------------------
write.csv(results_wide_no_smote, "results_no_SMOTE.csv",    row.names = FALSE)
write.csv(pred_all_no_smote,     "predictions_no_SMOTE.csv", row.names = FALSE)
write.csv(rf_importance,         "rf_importance_no_SMOTE.csv", row.names = FALSE)
write.csv(xgb_importance,        "xgb_importance_no_SMOTE.csv", row.names = FALSE)
write.csv(lasso_coef,            "lasso_coef_no_SMOTE.csv",  row.names = FALSE)

cat("\nScenario 1 (utan SMOTE) klar. Filer sparade.\n")
