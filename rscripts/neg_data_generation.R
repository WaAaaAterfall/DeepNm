rm(list = ls())
library(magrittr)
neg_data <- readRDS('./Tm_neg.rds')

neg_features <- data.frame(matrix(NA, nrow=neg_data$coverage %>% nrow, ncol=0))
feature_names <- paste0('mod_frac', seq(1, 41))
neg_mod_frac <- neg_data$mod_frac
neg_mod_frac[is.na(neg_mod_frac)] <- 0
colnames(neg_mod_frac) <- feature_names
neg_features <- cbind(neg_features, neg_mod_frac)
neg_coverage <- neg_data$coverage
feature_names <- paste0('coverage', seq(1, 41))
colnames(neg_coverage) <- feature_names
neg_coverage %>% is.na %>% any
neg_features <- cbind(neg_features, neg_coverage)

neg_deletion <- neg_data$deletion
feature_names <- paste0('deletion', seq(1, 41))
colnames(neg_deletion) <- feature_names
neg_deletion[is.na(neg_deletion)] <- 0
neg_features <- cbind(neg_features, neg_deletion)

neg_quality <- neg_data$quality
feature_names <- paste0('quality', seq(1, 41))
neg_quality[is.na(neg_quality)] <- -1
colnames(neg_quality) <- feature_names
neg_features <- cbind(neg_features, neg_quality)

neg_onehot <- neg_data$onehot
feature_names <- paste0('onehot', seq(1, 4004))
colnames(neg_onehot) <- feature_names
neg_features <- cbind(neg_features, neg_onehot)

write.csv(neg_features, './Tm_neg.csv')

