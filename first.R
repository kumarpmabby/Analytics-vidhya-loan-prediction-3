#kumar abhinav
#loading datasets and setting working directory
setwd("C:/Users/Lenovo/Downloads/tds/loan prediction 3/loan")
train <- read_csv("C:/Users/Lenovo/Downloads/tds/loan prediction 3/loan/train.csv")
test <- read_csv("C:/Users/Lenovo/Downloads/tds/loan prediction 3/loan/test.csv")

#univariate analysis
str(train)

#converting character to factors
train[sapply(train,is.character)]<-lapply(train[sapply(train,is.character)],as.factor)
test[sapply(test,is.character)]<-lapply(test[sapply(test,is.character)],as.factor)

#how many values missing
colSums(is.na(train))
colSums(is.na(test))


#imputations made for categorical values with mode ie missing value treatement
imputed_data<-impute(train,classes = list(factor=imputeMode()))
train<-imputed_data$data
imputed_data<-impute(test,classes = list(factor=imputeMode()))
test<-imputed_data$data

#imputations made for continuos values with mean ie missing value treatement
as.matrix(prop.table(table(train$Loan_Status)))
as.matrix(prop.table(table(train$Gender)))
as.matrix(prop.table(table(train$Dependents)))
#observing distributions of categorical values from above code
for(i in 1:ncol(train))
{
    train[is.na(train[,i]), i] <- mean(train[,i], na.rm = TRUE)
}
for(i in 1:ncol(test))
{
  test[is.na(test[,i]), i] <- mean(test[,i], na.rm = TRUE)
}

#changing data type of credit_History frm numeric to factor
train$Credit_History<-as.factor(train$Credit_History)
test$Credit_History<-as.factor(test$Credit_History)

#predictive modelling
table(train$Loan_Status)
#changing hot encoding of loan status
train$Loan_Status<- ifelse(train$Loan_Status=='N',0,1)
table(train$Loan_Status)
remove<- subset(train,select = -c(Loan_ID))

#decision tree solution
set.seed(333)
train.tree<-rpart(Loan_Status~ .,data=train,method ="class",control = rpart.control(minsplit =7,minbucket =30 ))
summary(train.tree)
train.tree<-rpart(Loan_Status~ .,data=remove,method ="class")
fancyRpartPlot(train.tree)
prediction_train <- predict(train.tree, newdata = train, type = "class")
prediction_test <- predict(train.tree, newdata = test, type = "class")
submit<-data.frame(Loan_ID=test$Loan_ID,Loan_Status=prediction_test)
write.csv(submit,file="dtreewithno_control_param.csv")
table(submit$Loan_Status)
train$Loan_Status<- ifelse(train$Loan_Status==0,'N','Y')

#decision trees solution with only credit_history and property area
set.seed(333)
fit<-rpart(Loan_Status~Credit_History+Property_Area,data=train,method ="class",control=rpart.control(minsplit=2, cp=0.005))
fancyRpartPlot(train.tree)

#analyze using confusion matrix
library(e1071)
confusionMatrix(prediction_train,train$Loan_Status)
#Confusion Matrix and Statistics

#Reference
#Prediction   N   Y
#N  82   7
#Y 110 415

#Accuracy : 0.8094          
#95% CI : (0.7761, 0.8398)
#No Information Rate : 0.6873          
#P-Value [Acc > NIR] : 6.062e-12       

#Kappa : 0.4808          
#Mcnemar's Test P-Value : < 2.2e-16       

#Sensitivity : 0.4271          
#Specificity : 0.9834          
#Pos Pred Value : 0.9213          
#Neg Pred Value : 0.7905          
#Prevalence : 0.3127          
#Detection Rate : 0.1336          
#Detection Prevalence : 0.1450          
#Balanced Accuracy : 0.7052          

#'Positive' Class : N  

#apply random forests
set.seed(415)
fit <- randomForest(as.factor(Loan_Status)~ Credit_History+ Property_Area+Gender+Dependents+Self_Employed,data=remove, importance=TRUE, ntree=2000)
varImpPlot(fit)
prediction_test<-predict(fit,test,type = "class")
prediction_train<-predict(fit,train,type = "class")
confusionMatrix(prediction_train,train$Loan_Status)
submit<-data.frame(Loan_ID=test$Loan_ID,Loan_Status=prediction_test)
write.csv(submit,file="randomforest_sol_with_no _featured_engineering.csv")









