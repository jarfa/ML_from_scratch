library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)

args = commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)

dat = read_csv(args[1])
plot_dir = args[2]
if(substring(plot_dir, nchar(plot_dir)) != "/"){
  plot_dir = paste0(plot_dir, "/")
}

test_metrics_dat = dat %>% 
  select(-train_norm_ll, -train_roc_auc) %>% 
  gather(key="metric", value="value", norm_ll, roc_auc) %>% 
  mutate(
    metric=ifelse(metric == "norm_ll", "Norm LogLoss", "ROC AUC"),
    target = as.factor(target)
    )

train_metrics_dat = dat %>%
  select(-norm_ll, -roc_auc) %>% 
  gather(key="metric", value="value", train_norm_ll, train_roc_auc) %>% 
  mutate(
    metric=ifelse(metric == "train_norm_ll", "Norm LogLoss", "ROC AUC"),
    target = as.factor(target)
  )

test_perf_by_model = test_metrics_dat %>% 
  ggplot(aes(x=model, y=value, color=source)) +
  geom_jitter(alpha=.7, width=.3) + 
  facet_wrap(~metric, scales="free") + theme_bw() + 
  ggtitle("Performance on Test Dataset") + 
  theme(axis.text.x = element_text(angle=-30, hjust=0))

train_perf_by_model= train_metrics_dat %>% 
  ggplot(aes(x=model, y=value, color=source)) +
  geom_jitter(alpha=.7, width=.3) + 
  facet_wrap(~metric, scales="free") + theme_bw() + 
  ggtitle("Performance on Train Dataset") + 
  theme(axis.text.x = element_text(angle=-30, hjust=0))

perf_by_target = test_metrics_dat %>% 
  ggplot(aes(x=target, y=value, color=model)) +
  geom_jitter(alpha=.7, width=.1) + 
  facet_wrap(~metric, scales="free_y") + theme_bw() +
  ggtitle("Performance on Test Dataset")

train_time = dat %>%
  ggplot(aes(x=model, y=train_time, color=source)) +
  geom_jitter(alpha=.7, width=.3) + 
  scale_y_log10("train time (seconds)") + theme_bw()

pred_time = dat %>% 
  ggplot(aes(x=model, y=pred_time, color=source)) +
  geom_jitter(alpha=.7, width=.3) + 
  scale_y_log10("prediction time (seconds)") +
  theme_bw()

plots = list(
  train_perf_by_model=train_perf_by_model,
  test_perf_by_model=test_perf_by_model,
  perf_by_target=perf_by_target,
  train_time=train_time,
  pred_time=pred_time
)
for(n in names(plots)){
  fname = paste0(plot_dir, n, ".png")
  ggsave(filename=fname, plot=plots[[n]], width=8.5, height=5.5)
}


correlations_across_models = function(metric){
  model_names = dat$model %>% unique
  by_model = do.call(cbind, lapply(model_names,
    function(m) dat[dat$model == m, metric])
    )
  correlations = cor(by_model)
  colnames(correlations) = model_names
  rownames(correlations) = model_names
  correlations %>% round(2)
}

cat("-----------")
cat("\n")
cat("Correlations on models' normalized LL\n")
correlations_across_models("norm_ll") %>% print
cat("Correlations on models' ROC AUC\n")
correlations_across_models("roc_auc") %>% print
cat("\n")
