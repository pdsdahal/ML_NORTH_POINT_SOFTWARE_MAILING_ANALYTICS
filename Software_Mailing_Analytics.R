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