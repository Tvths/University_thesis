# ============================================================
# Thesis_ML_class_weights.R
# Scenario 3: Maskininlärning med CLASS WEIGHTS
#
# FÖRBÄTTRINGAR vs originalversionen:
#   1) Tröskel optimeras per modell via Youdens J-statistik
#   2) XGBoost importance (xgb.importance) tillagd
#   3) LASSO-koefficienter (ej noll) visas och sparas
# ============================================================

library(tidymodels)
library(ranger)
library(xgboost)
library(glmnet)
library(dplyr)
library(ggplot2)
library(tidyr)
library(hardhat)
library(pROC)

tidymodels_prefer()

# ------------------------------------------------------------
# 1) Kontrollera input
# ------------------------------------------------------------
if (!exists("df")) {
  stop("Objektet 'df' finns inte. Kör först Thesis_hantera_NA.R.")
}

df <- df %>%
  select(-any_of(c("Citology", "Schiller", "Hinselmann", "Cytology"))) %>%
  mutate(
    Biopsy = case_when(
      Biopsy %in% c(1, "1", "Yes", "yes", "Y", "YES") ~ "Yes",
      Biopsy %in% c(0, "0", "No",  "no",  "N", "NO")  ~ "No",
      TRUE ~ as.character(Biopsy)
    ),
    Biopsy = factor(Biopsy, levels = c("No", "Yes"))
  ) %>%
  filter(!is.na(Biopsy))

cat("\nKlassfördelning:\n")
print(table(df$Biopsy))
print(prop.table(table(df$Biopsy)))

# ------------------------------------------------------------
# 2) Train/test-split
# ------------------------------------------------------------
set.seed(123)
data_split <- initial_split(df, prop = 0.80, strata = Biopsy)
train_data <- training(data_split)
test_data  <- testing(data_split)

cat("\nTrain:\n"); print(table(train_data$Biopsy))
cat("\nTest:\n");  print(table(test_data$Biopsy))

# ------------------------------------------------------------
# 3) Class weights – beräknas ENDAST från träningsdata
#    Formel: w_klass = n_total / (2 * n_klass)
#    Ger ungefär balanserad total vikt mellan klasser.
# ------------------------------------------------------------
n_no    <- sum(train_data$Biopsy == "No")
n_yes   <- sum(train_data$Biopsy == "Yes")
n_total <- nrow(train_data)

weight_no  <- n_total / (2 * n_no)
weight_yes <- n_total / (2 * n_yes)

cat("\nClass weights (från träningsdata):\n")
cat("No  =", round(weight_no, 4), "\n")
cat("Yes =", round(weight_yes, 4), "\n")
cat("Ratio Yes/No =", round(weight_yes / weight_no, 4), "\n")

train_data <- train_data %>%
  mutate(
    class_wts = if_else(Biopsy == "Yes", weight_yes, weight_no),
    class_wts = hardhat::importance_weights(class_wts)
  )

# Testdata viktas inte – men kolumnen behövs för att workflow ska fungera
test_data <- test_data %>%
  mutate(class_wts = hardhat::importance_weights(rep(1, n())))

# ------------------------------------------------------------
# 4) Recipe
# ------------------------------------------------------------
rec_class_weights <- recipe(Biopsy ~ ., data = train_data) %>%
  step_string2factor(all_nominal_predictors()) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# ------------------------------------------------------------
# 5) Metrics
# ------------------------------------------------------------
main_metrics <- metric_set(roc_auc, pr_auc, sens, spec,
                            precision, f_meas, mcc, brier_class)
select_metric <- "pr_auc"

# ------------------------------------------------------------
# 6) Tröskeloptimering via Youdens J
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
    val <- median(val, na.rm = TRUE)
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
# 7) Model specifications
# ------------------------------------------------------------
log_spec   <- logistic_reg() %>% set_engine("glm") %>% set_mode("classification")
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>% set_mode("classification")
rf_spec    <- rand_forest(trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_engine("ranger", importance = "permutation") %>% set_mode("classification")
xgb_spec   <- boost_tree(
  trees = tune(), tree_depth = tune(), learn_rate = tune(),
  loss_reduction = tune(), min_n = tune(), sample_size = tune()
) %>% set_engine("xgboost") %>% set_mode("classification")

# ------------------------------------------------------------
# 8) Workflows med case weights
# ------------------------------------------------------------
wf_log   <- workflow() %>% add_recipe(rec_class_weights) %>%
  add_model(log_spec) %>% add_case_weights(class_wts)
wf_lasso <- workflow() %>% add_recipe(rec_class_weights) %>%
  add_model(lasso_spec) %>% add_case_weights(class_wts)
wf_rf    <- workflow() %>% add_recipe(rec_class_weights) %>%
  add_model(rf_spec) %>% add_case_weights(class_wts)
wf_xgb   <- workflow() %>% add_recipe(rec_class_weights) %>%
  add_model(xgb_spec) %>% add_case_weights(class_wts)

# ------------------------------------------------------------
# 9) Korsvalidering och tuning
# ------------------------------------------------------------
set.seed(123)
folds <- vfold_cv(train_data, v = 5, strata = Biopsy)

set.seed(123)
fit_log <- fit(wf_log, data = train_data)

lasso_grid <- grid_regular(penalty(range = c(-5, 0)), levels = 30)
set.seed(123)
lasso_tune <- tune_grid(wf_lasso, resamples = folds, grid = lasso_grid,
                        metrics = metric_set(pr_auc, roc_auc),
                        control = control_grid(save_pred = TRUE))
best_lasso <- select_best(lasso_tune, metric = select_metric)
cat("\nBest LASSO parameters, class weights:\n"); print(best_lasso)
fit_lasso <- finalize_workflow(wf_lasso, best_lasso) %>% fit(data = train_data)

p_max <- ncol(train_data %>% select(-Biopsy, -class_wts))
rf_grid <- grid_latin_hypercube(
  mtry(range = c(1L, max(1L, p_max))),
  min_n(range = c(2L, 20L)),
  size = 10
)
set.seed(123)
rf_tune <- tune_grid(wf_rf, resamples = folds, grid = rf_grid,
                     metrics = metric_set(pr_auc, roc_auc),
                     control = control_grid(save_pred = TRUE))
best_rf <- select_best(rf_tune, metric = select_metric)
cat("\nBest RF parameters, class weights:\n"); print(best_rf)
fit_rf <- finalize_workflow(wf_rf, best_rf) %>% fit(data = train_data)

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
xgb_tune <- tune_grid(wf_xgb, resamples = folds, grid = xgb_grid,
                      metrics = metric_set(pr_auc, roc_auc),
                      control = control_grid(save_pred = TRUE))
best_xgb <- select_best(xgb_tune, metric = select_metric)
cat("\nBest XGBoost parameters, class weights:\n"); print(best_xgb)
fit_xgb <- finalize_workflow(wf_xgb, best_xgb) %>% fit(data = train_data)

# ------------------------------------------------------------
# 10) Prediktion med Youdens J-tröskel
# ------------------------------------------------------------
cat("\n--- Youdens J-trösklar (class weights) ---\n")
pred_all_cw <- bind_rows(
  get_predictions(fit_log,   test_data, "Weighted Logistic Regression"),
  get_predictions(fit_lasso, test_data, "Weighted LASSO Logistic Regression"),
  get_predictions(fit_rf,    test_data, "Weighted Random Forest"),
  get_predictions(fit_xgb,   test_data, "Weighted XGBoost")
)

results_cw <- make_results(pred_all_cw)
results_wide_cw <- results_cw %>%
  select(model, .metric, .estimate) %>%
  pivot_wider(names_from = .metric, values_from = .estimate)

thresholds_cw <- pred_all_cw %>%
  group_by(model) %>%
  summarise(threshold = first(threshold), .groups = "drop")

results_wide_cw <- results_wide_cw %>%
  left_join(thresholds_cw, by = "model") %>%
  arrange(desc(pr_auc))

cat("\n================ Resultat: CLASS WEIGHTS ================\n")
print(results_wide_cw)

cat("\n================ Confusion matrices: CLASS WEIGHTS ================\n")
print(pred_all_cw %>%
        group_by(model) %>%
        conf_mat(truth = Biopsy, estimate = .pred_class))

# ------------------------------------------------------------
# 11) ROC- och PR-kurvor
# ------------------------------------------------------------
roc_all_cw <- pred_all_cw %>%
  group_by(model) %>%
  roc_curve(truth = Biopsy, .pred_Yes, event_level = "second")

ggplot(roc_all_cw, aes(x = 1 - specificity, y = sensitivity, color = model)) +
  geom_path(linewidth = 1) + geom_abline(linetype = 2) +
  labs(title = "ROC-kurvor, class weights",
       x = "1 - Specificitet", y = "Sensitivitet", color = "Modell")

pr_all_cw <- pred_all_cw %>%
  group_by(model) %>%
  pr_curve(truth = Biopsy, .pred_Yes, event_level = "second")

ggplot(pr_all_cw, aes(x = recall, y = precision, color = model)) +
  geom_path(linewidth = 1) +
  labs(title = "Precision-Recall-kurvor, class weights",
       x = "Recall", y = "Precision", color = "Modell")

# ------------------------------------------------------------
# 12) Variabelviktighet – Random Forest
# ------------------------------------------------------------
rf_engine_cw <- extract_fit_engine(fit_rf)
rf_importance_cw <- tibble(
  variable   = names(rf_engine_cw$variable.importance),
  importance = as.numeric(rf_engine_cw$variable.importance)
) %>% arrange(desc(importance))

cat("\nTop 20 variable importance, RF, class weights:\n")
print(rf_importance_cw %>% slice_head(n = 20))

ggplot(rf_importance_cw %>% slice_head(n = 20),
       aes(x = reorder(variable, importance), y = importance)) +
  geom_col(fill = "#D7191C", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest: Permutationsviktighet (class weights)",
       x = NULL, y = "Viktighet") +
  theme_minimal()

# ------------------------------------------------------------
# 13) Variabelviktighet – XGBoost
# ------------------------------------------------------------
xgb_engine_cw <- extract_fit_engine(fit_xgb)
xgb_importance_cw <- xgb.importance(model = xgb_engine_cw) %>%
  as_tibble() %>%
  arrange(desc(Gain))

cat("\nTop 20 variable importance, XGBoost (Gain), class weights:\n")
print(xgb_importance_cw %>% slice_head(n = 20))

ggplot(xgb_importance_cw %>% slice_head(n = 20),
       aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "#1A9641", alpha = 0.8) +
  coord_flip() +
  labs(title = "XGBoost: Feature importance – Gain (class weights)",
       x = NULL, y = "Gain") +
  theme_minimal()

# ------------------------------------------------------------
# 14) Variabelviktighet – LASSO-koefficienter
# ------------------------------------------------------------
lasso_engine_cw <- extract_fit_engine(fit_lasso)
lasso_coef_cw <- coef(lasso_engine_cw, s = best_lasso$penalty) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column("variable") %>%
  rename(coefficient = 2) %>%
  filter(variable != "(Intercept)", coefficient != 0) %>%
  mutate(direction = if_else(coefficient > 0,
                             "Ökad risk (+)", "Minskad risk (−)")) %>%
  arrange(desc(abs(coefficient)))

cat("\nLASSO: Antal variabler ≠ 0 (class weights):", nrow(lasso_coef_cw), "\n")
print(lasso_coef_cw)

ggplot(lasso_coef_cw,
       aes(x = reorder(variable, abs(coefficient)),
           y = coefficient, fill = direction)) +
  geom_col(alpha = 0.85) +
  coord_flip() +
  scale_fill_manual(values = c("Ökad risk (+)"    = "#E05A5A",
                               "Minskad risk (−)" = "#4DBBAE")) +
  labs(title    = "LASSO: Kvarstående koefficienter (class weights)",
       subtitle = "Variabler krympta till 0 visas ej",
       x = NULL, y = "Koefficient (standardiserad skala)", fill = NULL) +
  theme_minimal()

# ------------------------------------------------------------
# 15) Spara
# ------------------------------------------------------------
write.csv(results_wide_cw,   "results_class_weights.csv",        row.names = FALSE)
write.csv(pred_all_cw,       "predictions_class_weights.csv",    row.names = FALSE)
write.csv(rf_importance_cw,  "rf_importance_class_weights.csv",  row.names = FALSE)
write.csv(xgb_importance_cw, "xgb_importance_class_weights.csv", row.names = FALSE)
write.csv(lasso_coef_cw,     "lasso_coef_class_weights.csv",     row.names = FALSE)

cat("\nScenario 3 (class weights) klar. Filer sparade.\n")
