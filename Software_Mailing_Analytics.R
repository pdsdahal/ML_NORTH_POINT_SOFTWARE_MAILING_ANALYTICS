# working directory
#getwd()
#setwd("/Users/sdahal/GitRepos/ML_NORTH_POINT_SOFTWARE_MAILING_ANALYTICS")
library(dplyr)
library(ggplot2)
library(tidyr)
library(psych)
library(caret)
library(corrplot)
library(car)
library(splitTools)
library(pROC)
library(randomForest)
# load the dataset
software_mailing_list_data <- read.csv("dataset/Software_Mailing_List.csv")
# structure
str(software_mailing_list_data)
#View(software_mailing_list_data)

# Numerical Data
numeric_cols <- c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")
numeric_datas <- software_mailing_list_data[, numeric_cols]

# Summary statistics
summary(numeric_datas)

# 3.1.3. Analysis of Data Recency and Customer Value
# Histograms for numeric variables
par(mfrow = c(2, 2))  # 2x2 grid for plots
for (col in numeric_cols) {
  hist(software_mailing_list_data[[col]], main = paste("Distribution of", col), xlab = col, col = "lightblue")
}

# relation ship between Spending and last_update_days_ago 
pairs.panels(software_mailing_list_data[, c("last_update_days_ago", "Spending")])

#3.1.4. Impact of Web Channel on Spending
# Boxplot: Spending by Web order (for purchasers)
ggplot(software_mailing_list_data %>% filter(Purchase == 1), aes(x = factor(`Web.order`), y = Spending)) +
  geom_boxplot(fill = "lightyellow") +
  labs(title = "Spending by Web Order (Purchasers)", x = "Web.Order", y = "Spending") +
  theme_minimal()

#3.1.5. Influence of Past Purchase Frequency on Campaign Response
# Boxplot: Freq by Purchase
ggplot(software_mailing_list_data, aes(x = factor(Purchase), y = Freq)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Frequency by Purchase", x = "Purchase", y = "Frequency") +
  theme_minimal()

#3.1.6. Categorical Data
#  categorical columns
# binary variables (treated as categorical for analysis)
categorical_cols <- c("US", "source_a", "source_b", "source_c", "source_d", "source_e", 
                      "source_m", "source_o", "source_h", "source_r", "source_s", 
                      "source_t", "source_u", "source_p", "source_x", "source_w", 
                      "Web.order", "Gender.male", "Address_is_res", "Purchase")

# Frequency tables with percentages
categorical_frq_results <- data.frame()

for (col in categorical_cols) {
  pct <- round(prop.table(table(software_mailing_list_data[[col]])) * 100, 2)
  categorical_frq_results <- rbind(categorical_frq_results, data.frame(
    Variable_Name = col,
    Value_0 = ifelse(!is.na(pct["0"]), pct["0"], 0),
    Value_1 = ifelse(!is.na(pct["1"]), pct["1"], 0)
  ))
}
print(categorical_frq_results)

# Source List Distribution Analysis
source_cols <- c("source_a", "source_b", "source_c", "source_d", "source_e",
                 "source_m", "source_o", "source_h", "source_r", "source_s",
                 "source_t", "source_u", "source_p", "source_x", "source_w")

# 5 rows x 3 columns
par(mfrow = c(5, 3), mar = c(4, 4, 2, 1))  

# Loop through sources and plot barplots
for (col in source_cols) {
  counts <- table(software_mailing_list_data[[col]])
  barplot(counts, main = col, col = c("lightblue", "lightgreen"), 
          ylim = c(0, max(counts) * 1.1), ylab = "Count")
}
# Distribution of Key Customer Attributes and Campaign Response
categorical_cols_except_sources <- c("US", "Web.order", "Gender.male", "Address_is_res", "Purchase")
par(mfrow = c(2, 3), mar = c(4, 4, 2, 1))

for (col in categorical_cols_except_sources) {
  counts <- table(software_mailing_list_data[[col]])
  barplot(counts,
          main = col,
          col = c("lightblue", "lightgreen"),
          ylim = c(0, max(counts) * 1.1),
          ylab = "Count")
}
##Data Preprocessing###############################
# checking missing value
missing_summary <- colSums(is.na(software_mailing_list_data))
data.frame(Missing_Count = missing_summary, 
           Missing_Percent = paste0(round((missing_summary / nrow(software_mailing_list_data)) * 100, 2), "%"))

# Check for empty strings
sum(software_mailing_list_data == "", na.rm = TRUE)

# checking zeros entire data set except categorical variables 
numeric_cols <- c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")
zeros_summary <- colSums(software_mailing_list_data[numeric_cols] == 0, na.rm = TRUE)
# summary with percent
zeros_df <- data.frame(
  Column = names(zeros_summary),
  Zero_Count = zeros_summary,
  Zero_Percent = paste0(round((zeros_summary / nrow(software_mailing_list_data)) * 100, 2), "%")
)
zeros_df

#  duplicates check 
sum(duplicated(software_mailing_list_data$sequence_number))
######################## Outliers Detection
numeric_cols <- c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")
# Function to calculate outlier count using IQR
find_outliers <- function(x) {
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr <- q3 - q1
  sum(x < (q1 - 1.5*iqr) | x > (q3 + 1.5*iqr), na.rm = TRUE)
}
# apply function to all numeric columns
outlier_counts <- sapply(software_mailing_list_data[numeric_cols], find_outliers)

# create summary table without duplicating column names
outliers_df <- data.frame(
  Variable = names(outlier_counts),
  Outlier_Count = as.numeric(outlier_counts),
  Min_Value = round(sapply(software_mailing_list_data[numeric_cols], function(x) min(x, na.rm = TRUE)), 2),
  Max_Value = round(sapply(software_mailing_list_data[numeric_cols], function(x) max(x, na.rm = TRUE)), 2),
  Percentage = paste0(round(as.numeric(outlier_counts) / nrow(software_mailing_list_data) * 100, 2), "%"),
  row.names = NULL
)
print(outliers_df)

# Predictor Analysis and Relevancy
# Correlation Analysis for Numeric Predictors
numeric_data <- software_mailing_list_data[, c("Freq", "last_update_days_ago", "X1st_update_days_ago", "Spending")]

# Calculate correlation matrix
cor_matrix <- cor(numeric_data, use = "complete.obs")
cor_matrix

options(repr.plot.width = 16, repr.plot.height = 14)  
par(mfrow = c(1, 1), mar = c(2, 2, 2, 2))

# Increase text size
corrplot(cor_matrix,
         method = "color",
         type = "upper",
         tl.cex = 1.0,        # larger variable names
         tl.col = "black",
         addCoef.col = "black",
         number.cex = 1.2,    # larger correlation numbers
         col = colorRampPalette(c("blue", "white", "red"))(200))

#Predictor Relevancy
#Frequency Variable Relevance Analysis Using Logistic Regression 
software_mailing_list_data_Relevancy <- software_mailing_list_data
software_mailing_list_data_Relevancy$Freq_scaled <- scale(software_mailing_list_data_Relevancy$Freq)
model_scaled <- glm(Purchase ~ Freq_scaled, 
                    data = software_mailing_list_data_Relevancy, 
                    family = binomial)
summary(model_scaled)

############## categorical Relevency

# Chi-squared tests for source variables
source_cols <- c("source_a", "source_b", "source_c", "source_d", "source_e", 
                 "source_m", "source_o", "source_h", "source_r", "source_s", 
                 "source_t", "source_u", "source_p", "source_x", "source_w")

results_chi_cat <- lapply(source_cols, function(col) {
  test <- chisq.test(software_mailing_list_data[[col]], 
                     software_mailing_list_data$Purchase)
  
  data.frame(Source = col,
             Chi_Sq = as.numeric(test$statistic),
             df = test$parameter,
             P_Value = test$p.value)
})

# Combine into one data frame
results_chi_cat <- do.call(rbind, results_chi_cat)

# Reset row names to simple numbers
rownames(results_chi_cat) <- NULL

# Sort by p-value
results_chi_cat <- results_chi_cat[order(results_chi_cat$P_Value), ]

# Print results
print("Chi-squared Test Results for Source Variables")
print(results_chi_cat)

# Chi-squared test for Web.order and Purchase
web_chi <- chisq.test(software_mailing_list_data$Web.order, software_mailing_list_data$Purchase)
print("Chi-squared test for Web.order and Purchase:")
print(web_chi)

# Chi-squared test for us and Purchase
chisq.test(software_mailing_list_data$US, software_mailing_list_data$Purchase)

# Chi-squared test for Gender.male and Purchase
chisq.test(software_mailing_list_data$Gender.male, software_mailing_list_data$Purchase)

# Chi-squared test for Address_is_res and Purchase
chisq.test(software_mailing_list_data$Address_is_res, software_mailing_list_data$Purchase)

################ 5.1 Dimension Reduction Analysis

#feature engineering (proffessor fedback)
software_mailing_with_active <- software_mailing_list_data
software_mailing_with_active$active_sources <- rowSums(software_mailing_with_active[, source_cols])

#  distribution of active_sources
table(software_mailing_with_active$active_sources)

# Subset the 0-source records from the new dataframe
no_source_records <- software_mailing_with_active[software_mailing_with_active$active_sources == 0, ]

# Check purchase distribution
table(no_source_records$Purchase)

# Now create the new variable for source_category
software_mailing_list_data$source_category <- apply(
  software_mailing_list_data[source_cols], 1, function(row) {
    active <- names(row)[row == 1]        
    if (length(active) == 0) {
      return("none")                     
    } else {
      return(sub("source_", "", active))  
    }
  }
)
# Convert to factor
software_mailing_list_data$source_category <- factor(software_mailing_list_data$source_category)

# Verify result
table(software_mailing_list_data$source_category)

#Detect Data Quality Issue
# Find records where Purchase = 0 and Spending = 1
original_count <- sum(software_mailing_list_data$Purchase == 0 & software_mailing_list_data$Spending == 1)
original_count

# Will change the Purchase variable from 0 to 1 for records where:
#Current Purchase = 0 (no purchase)
#Spending = 1 (but they spent $1)
software_mailing_list_data$Purchase[software_mailing_list_data$Purchase == 0 & software_mailing_list_data$Spending == 1] <- 1

# Verify
new_count <- sum(software_mailing_list_data$Purchase == 0 & software_mailing_list_data$Spending == 1)
new_count
#View(software_mailing_list_data)
#need to make copy 
software_data_classification <- software_mailing_list_data

# removed all the sources variables only keept source_category 
software_data_classification <- software_data_classification[, c("US","Freq","last_update_days_ago","X1st_update_days_ago",
                                                                 "Web.order","Gender.male","Address_is_res","Purchase",
                                                                 "Spending","source_category")]
##########Feature Importance from Random Forest
remove_vars <- c("sequence_number", "X1st_update_days_ago", "Spending")
software_rf_feature_imp <- software_data_classification
software_rf_feature_imp <- software_rf_feature_imp[, !(names(software_rf_feature_imp) %in% remove_vars)]
software_rf_feature_imp$Purchase <- as.factor(software_rf_feature_imp$Purchase)
software_rf_feature_imp$Address_is_res <- as.factor(software_rf_feature_imp$Address_is_res)
software_rf_feature_imp$Web.order      <- as.factor(software_rf_feature_imp$Web.order)
software_rf_feature_imp$US             <- as.factor(software_rf_feature_imp$US)

#str(software_rf_feature_imp)
library(randomForest)
# Train Random Forest model
set.seed(2025)
rf_model <- randomForest(Purchase ~ ., 
                         data = software_rf_feature_imp, 
                         importance = TRUE)
rf_importance <- randomForest::importance(rf_model)
# Convert to data frame
rf_importance_df <- data.frame(
  Variable = rownames(rf_importance),
  MeanDecreaseAccuracy = rf_importance[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini     = rf_importance[, "MeanDecreaseGini"]
)
# Sort by MeanDecreaseAccuracy
rf_importance_df <- rf_importance_df[order(rf_importance_df$MeanDecreaseAccuracy, decreasing = TRUE), ]
rownames(rf_importance_df) <- NULL
print(rf_importance_df)

#View(software_data_classification)
# now final selected features for classifications 
#US
# Selected predictors based on correlation & Feature Importance from Random Forest
selected_features_classification <- c("Freq","source_category","last_update_days_ago",
                                      "Address_is_res","Web.order","Purchase")

#  predictors & target variable for classification
software_data_classification <- software_data_classification[, selected_features_classification]
#View(software_data_classification)
#str(software_data_classification)

# #############Data Partition 
strata_factor <- interaction(
  software_data_classification$Purchase,
  drop = TRUE
)
# perform a stratified 60/20/20 split
set.seed(2025)
splits <- partition(
  y = strata_factor, 
  p = c(train = 0.6, val = 0.2, test = 0.2),
  type = "stratified"
)

# Extract partitions
train_data <- software_data_classification[splits$train, ]
val_data   <- software_data_classification[splits$val, ]
test_data  <- software_data_classification[splits$test, ]

###################### let create summary
# Partition sizes
train_n <- nrow(train_data)
val_n   <- nrow(val_data)
test_n  <- nrow(test_data)

# Total records
total_n <- nrow(software_data_classification)

# Create summary table
partition_summary <- data.frame(
  Partition   = c("Training", "Validation", "Test"),
  Records     = c(train_n, val_n, test_n),
  Percentage  = round(c(train_n, val_n, test_n) / total_n * 100, 2)
)

print(partition_summary)

# partiton distribution for categorical predictors
categorical_vars <- c("Purchase","Web.order","Address_is_res")
partition_distribution_summary <- data.frame()

for (var in categorical_vars) {
  train_tab <- prop.table(table(train_data[[var]])) * 100
  val_tab   <- prop.table(table(val_data[[var]])) * 100
  test_tab  <- prop.table(table(test_data[[var]])) * 100
  
  for (lev in names(train_tab)) {
    temp_df <- data.frame(
      VariableName  = var,
      Level         = lev,
      Train_Percent = round(train_tab[lev],2),
      Val_Percent   = round(val_tab[lev],2),
      Test_Percent  = round(test_tab[lev],2)
    )
    partition_distribution_summary <- rbind(partition_distribution_summary, temp_df)
  }
}

print(partition_distribution_summary)

# Check overlap between splits
length(intersect(splits$train, splits$test))
length(intersect(splits$train, splits$val)) 
length(intersect(splits$val, splits$test))

# Check coverage
length(unique(c(splits$train, splits$val, splits$test))) == nrow(software_data_classification)

# Decided to use Logistic Regression, Random Forest, K-means Clustering
# predictors & target variable for classification

#  factors common method
factorize_predictors <- function(df) {
  df$Address_is_res <- as.factor(df$Address_is_res)
  df$Web.order      <- as.factor(df$Web.order)
  return(df)
}

# Apply function
X_train_cat <- factorize_predictors(train_data[, !names(train_data) %in% "Purchase"])
X_val_cat   <- factorize_predictors(val_data[, !names(val_data) %in% "Purchase"])
X_test_cat  <- factorize_predictors(test_data[, !names(test_data) %in% "Purchase"])

# Targets as factors (with explicit levels for consistency)
y_train_cat <- factor(train_data$Purchase, levels = c(0, 1))
y_val_cat   <- factor(val_data$Purchase, levels = c(0, 1))
y_test_cat  <- factor(test_data$Purchase, levels = c(0, 1))