# load necessary libraries
install.packages("tree")
install.packages("randomForest")
install.packages("caret")
install.packages("pastecs")
install.packages("ggplot2")

# load the library packages
library(tree)
library(randomForest)
library(caret)
library(pastecs)
library(ggplot2)

# load stroke dataset
data <- read.csv("/Users/yoycimedina/Desktop/Documents/Academics/BUS _315/Midterm/Midterm_R_PT2/stroke_data.csv")

#### Data Exploration ####

# investigate the data
View(data)
head(data) 
tail(data)

# Check for missing values in the dataset
is.na(data) #finds True False missing values
sum(is.na(data)) #sum up the number of missing values in the data set
sum(is.na(data$bmi)) #sum up the number of missing values for a specific attribute 

# summary statistics
summary(data)
str(data) # structure of data
pastecs::stat.desc(data) #descriptive stats

# investigate numerical data attributes
cor(data$avg_glucose_level, data$bmi)
cor(data$age, data$avg_glucose_level)
lapply(data$bmi[1:4], mean)
lapply(data$age[1:4], mean)
lapply(data$avg_glucose_level[1:4], mean)

# visual investigation of the data
plot(data$age)
plot(data$age, data$avg_glucose_level)
plot(data$age, data$bmi)

hist(data$age, main="Distribution of Age", xlab="Age")
hist(data$avg_glucose_level, main="Distribution of Average Glucose Level", xlab="Average Glucose Level")
hist(data$bmi, main="Distribution of BMI", xlab="BMI")

qplot(age, avg_glucose_level, data = data, colour = stroke) + ggtitle("Age vs. Avg Glucose Level by Stroke Status")
qplot(age, bmi, data = data, colour = stroke) + ggtitle("Age vs. BMI by Stroke Status")
qplot(bmi, avg_glucose_level, data = data, colour = stroke) + ggtitle("BMI vs. Avg Glucose Level by Stroke Status")

# detailed descriptive analysis using pastecs
stat.desc(data)

#### Preprocessing ####

# drop the 'id' column
data <- data[-which(names(data) == "id")]

#finds True False missing values in bmi column
sum(is.na(data))
sum(is.na(data$bmi))
is.na(data$bmi)
data = na.omit(data)

# convert 'bmi' to numeric
data$bmi <- as.numeric(data$bmi)

# convert attributes to appropriate data types
data$gender = as.factor(data$gender)
data$ever_married = as.factor(data$ever_married)
data$work_type = as.factor(data$work_type)
data$Residence_type = as.factor(data$Residence_type)
data$smoking_status = as.factor(data$smoking_status)
data$hypertension = ifelse(data$hypertension==0,"No hypertension",'Has hypertension')
data$hypertension = as.factor(data$hypertension)
data$heart_disease = ifelse(data$heart_disease==0,"No heart diseases",'Has heart diseases')
data$heart_disease = as.factor(data$heart_disease)
data$bmi = as.numeric(data$bmi)
data$stroke = ifelse(data$stroke==0,"No stroke",'Stroke')
data$stroke = as.factor(data$stroke)


#### Create Train and Test Sets ####

set.seed(123)
train_index <- sample(1:nrow(data), .8*nrow(data))
train_data <- data[train_index, ] # observations, 364
test_data <- data[-train_index, ] # observations, 91
stroke_test <- data$stroke[-train_index]


#### Model Building & Evaluation ####

# train a decision tree model using the 'tree' package
dtree <- tree(stroke ~ ., data=train_data)
summary(dtree) # misclassification error rate 18.68%
# visualize the tree model
plot(dtree)
text(dtree)

dtree_tested = predict(dtree, test_data, type = "class")
table(dtree_tested, stroke_test)
accuracy_tested = (29+37)/91
accuracy_tested
print(paste("Decision Tree Model Accuracy: ", accuracy_tested))

# cross-validate the decision tree to determine optimal size
cv_tree <- cv.tree(dtree, FUN=prune.misclass)
optimal_size <- which.min(cv_tree$dev) # Find the size with the lowest misclassification error

# prune the tree based on the optimal size
pruned_dtree <- prune.misclass(dtree, best = optimal_size)
summary(pruned_dtree) # misclassification error rate 20.05%
# visualize the pruned tree
plot(pruned_dtree)
text(pruned_dtree)

# test the pruned decision tree model
pruned_dtree_predictions <- predict(pruned_dtree, test_data, type="class")
table(pruned_dtree_predictions, stroke_test)
accuracy_tested = (27+37)/91
accuracy_tested 
print(paste("Pruned Tree Model Accuracy: ", accuracy_tested))

# train a randomForest model
dtree_rf = randomForest::randomForest(stroke~., train_data)
# test the random forest model
dtree_rf_tested = predict(dtree_rf, test_data, type = "class")
table(dtree_rf_tested, stroke_test)
accuracy_rf = (32+30)/91
accuracy_rf
print(paste("Random Forest Model Accuracy: ", accuracy_rf)) 

