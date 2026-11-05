# ============================================================
# GROUP CASE PROJECT: SOCIAL MEDIA & MENTAL HEALTH
# Steps 1, 2, & 3: Load, Clean, Explore, and Test
# ============================================================

# --- 1. Load Libraries ---
library(tidyverse)
library(caret)
library(readxl)
library(corrplot)
library(car)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
# Note: Make sure the excel file is in your working directory!
# If you are using the direct path, change it back to: 
# df <- read_excel("C:/Users/alee8/Downloads/Social Media and Mental Health.xlsx")
df <- read_excel("C:/Users/alee8/Downloads/Social Media and Mental Health.xlsx")

# ============================================================
# STEP 2: CLEAN & PRE-PROCESS
# ============================================================

# 2.1 Rename columns to clean, readable names
df_clean <- df %>%
  rename(
    timestamp         = `Timestamp`,
    age               = `1. What is your age?`,
    gender            = `2. Gender`,
    relationship      = `3. Relationship Status`,
    occupation        = `4. Occupation Status`,
    organization      = `5. What type of organizations are you affiliated with?`,
    uses_social       = `6. Do you use social media?`,
    platforms         = `7. What social media platforms do you commonly use?`,
    time_spent        = `8. What is the average time you spend on social media every day?`,
    no_purpose        = `9. How often do you find yourself using Social media without a specific purpose?`,
    distracted        = `10. How often do you get distracted by Social media when you are busy doing something?`,
    restless          = `11. Do you feel restless if you haven't used Social media in a while?`,
    easily_distracted = `12. On a scale of 1 to 5, how easily distracted are you?`,
    bothered_worries  = `13. On a scale of 1 to 5, how much are you bothered by worries?`,
    concentration     = `14. Do you find it difficult to concentrate on things?`,
    compares          = `15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?`,
    comparison_feel   = `16. Following the previous question, how do you feel about these comparisons, generally speaking?`,
    validation        = `17. How often do you look to seek validation from features of social media?`,
    depressed         = `18. How often do you feel depressed or down?`,
    interest_fluct    = `19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?`,
    sleep_issues      = `20. On a scale of 1 to 5, how often do you face issues regarding sleep?`
  ) %>%
  select(-timestamp)

# 2.2 Clean messy categorical values (Gender typos)
df_clean <- df_clean %>%
  mutate(
    gender = str_trim(str_to_lower(gender)),
    gender = case_when(
      gender == "male" ~ "Male",
      gender == "female" ~ "Female",
      gender %in% c("nonbinary", "non-binary", "nb", "non binary") ~ "Nonbinary",
      gender == "trans" ~ "Trans",
      gender == "unsure" ~ "Unsure",
      TRUE ~ "Other" 
    ),
    gender = as.factor(gender)
  )

# 2.3 Convert remaining character columns to factors
df_clean <- df_clean %>%
  mutate(
    relationship  = as.factor(relationship),
    occupation    = as.factor(occupation),
    organization  = as.factor(organization),
    uses_social   = as.factor(uses_social)
  )

# 2.4 Handle Age (Drop the erroneous 91 age outlier)
df_clean <- df_clean %>%
  mutate(age = as.numeric(age)) %>%
  filter(age < 90) 

# 2.5 Create the Dependent Variable (Binary Depression)
df_clean <- df_clean %>%
  mutate(
    depressed_binary = ifelse(depressed >= 4, "High", "Low"),
    depressed_binary = factor(depressed_binary, levels = c("Low", "High"))
  )

# 2.6 Split platforms into binary columns (1 = uses, 0 = doesn't use)
platform_list <- c("Facebook", "Twitter", "Instagram", "YouTube", "Snapchat", 
                   "Discord", "Reddit", "Pinterest", "TikTok")

for (p in platform_list) {
  df_clean[[paste0("uses_", p)]] <- as.integer(str_detect(df_clean$platforms, p))
}
df_clean <- df_clean %>% select(-platforms)

# 2.7 Create the Validation Column (70% train, 30% test)
set.seed(123) 
train_idx <- createDataPartition(df_clean$depressed_binary, p = 0.7, list = FALSE)
df_clean$validation <- "test"
df_clean$validation[train_idx] <- "train"
df_clean$validation <- as.factor(df_clean$validation)

# Export the clean dataset for your team
write.csv(df_clean, "clean_smmh_data.csv", row.names = FALSE)


# ============================================================
# STEP 3: EXPLORE & HYPOTHESIS TESTS
# ============================================================

# 3.1 Hypothesis Tests (Categorical vs. Dependent Variable)
print("--- Chi-Square Tests ---")
chisq.test(df_clean$gender, df_clean$depressed_binary)
chisq.test(df_clean$relationship, df_clean$depressed_binary)
chisq.test(df_clean$occupation, df_clean$depressed_binary)
chisq.test(df_clean$uses_social, df_clean$depressed_binary)

# 3.2 Hypothesis Tests (Numeric vs. Dependent Variable)
print("--- T-Tests ---")
t.test(age ~ depressed_binary, data = df_clean)

# 3.3 Correlation Tests (Survey Questions vs. Depression)
print("--- Correlation Tests ---")
cor.test(df_clean$no_purpose, df_clean$depressed) 
cor.test(df_clean$restless, df_clean$depressed)
cor.test(df_clean$sleep_issues, df_clean$depressed)
cor.test(df_clean$comparison_feel, df_clean$depressed)

# 3.4 Multicollinearity Check
numeric_vars <- df_clean %>% select(where(is.numeric))
cor_matrix <- cor(numeric_vars, use = "complete.obs")
print("--- Correlation Matrix ---")
print(round(cor_matrix, 2))

# 3.5 Quick Distribution Plots for Report
# (You can save these to include in your report)
plot1 <- ggplot(df_clean, aes(x = depressed)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Distribution of Depression Levels", x = "Depression Score (1-5)", y = "Count")

plot2 <- ggplot(df_clean, aes(x = depressed_binary, y = age, fill = depressed_binary)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Age Distribution by Depression Level", x = "Depression Level", y = "Age")

print(plot1)
print(plot2)


# ============================================================
# STEP 4: CHOOSE A MODEL (Logistic Regression)
# ============================================================

# Group rare categories so the model doesn't crash
df_clean <- df_clean %>%
  mutate(
    gender = fct_lump(gender, n = 2), # Keeps Male/Female, lumps rest to "Other"
    relationship = fct_lump(relationship, n = 3), # Keeps top 3, lumps rest
    occupation = fct_lump(occupation, n = 2), # Keeps top 2, lumps rest
    organization = fct_lump(organization, n = 3) # Keeps top 3, lumps rest
  )


# 1. Build the full model (note we left out "validation")
model_full <- glm(depressed_binary ~ age + gender + relationship + occupation + 
                    time_spent + no_purpose + distracted + restless + 
                    easily_distracted + bothered_worries + concentration + 
                    compares + comparison_feel + interest_fluct + 
                    sleep_issues + uses_Facebook + uses_Twitter + uses_Instagram + 
                    uses_YouTube + uses_Snapchat + uses_Discord + uses_Reddit + 
                    uses_Pinterest + uses_TikTok, 
                  data = df_clean %>% filter(validation == "train"), 
                  family = "binomial")

# 2. Run the stepwise selection
model_reduced <- step(model_full, direction = "backward")

# 3. Print the final result
summary(model_reduced)


# ============================================================
# STEP 5: EVALUATE PERFORMANCE
# ============================================================

# 1. Make predictions on the TEST data
# type = "response" gives us probabilities between 0 and 1
predictions <- predict(model_reduced, 
                       newdata = df_clean %>% filter(validation == "test"), 
                       type = "response")

# 2. Convert probabilities to "High" or "Low" (threshold = 0.5)
predicted_class <- ifelse(predictions > 0.5, "High", "Low")

# 3. Create a confusion matrix to see accuracy
actual_class <- df_clean %>% filter(validation == "test") %>% pull(depressed_binary)

confusionMatrix(factor(predicted_class, levels = c("Low", "High")), 
                factor(actual_class, levels = c("Low", "High")))


# ============================================================
# STEP 6: DEPLOY BEST MODEL
# ============================================================

# 1. Run predictions on the ENTIRE dataset
df_clean$predicted_prob <- predict(model_reduced, newdata = df_clean, type = "response")

# Convert probabilities to "High" or "Low" predictions
df_clean$predicted_class <- ifelse(df_clean$predicted_prob > 0.5, "High", "Low")
df_clean$predicted_class <- factor(df_clean$predicted_class, levels = c("Low", "High"))

# 2. Highlight a few predictions to show in the report
# Let's look at 5 random people to see their actual vs predicted
set.seed(42)
sample_indices <- sample(1:nrow(df_clean), 5)

prediction_sample <- df_clean %>%
  select(age, bothered_worries, compares, interest_fluct, 
         depressed_binary, predicted_prob, predicted_class) %>%
  slice(sample_indices)

print("--- Sample Predictions ---")
print(prediction_sample)

# 3. See the overall accuracy on the FULL dataset
confusionMatrix(df_clean$predicted_class, df_clean$depressed_binary)