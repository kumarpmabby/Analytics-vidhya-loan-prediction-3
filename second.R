#kumar abhinav
#loading datasets and setting working directory
setwd("C:/Users/Lenovo/Downloads/tds/loan prediction 3/loan")
train <- read_csv("C:/Users/Lenovo/Downloads/tds/loan prediction 3/loan/train.csv")
test <- read_csv("C:/Users/Lenovo/Downloads/tds/loan prediction 3/loan/test.csv")


# load required packages for fancy decision tree plotting and random forest
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(randomForest)
library(party)

#multi variate analysis
#Applicant Income and 7. CoApplicant Income.
par(mfrow=c(1,2))
boxplot(train$ApplicantIncome,train$CoapplicantIncome,names = c("App income","coapp income"),main="train")
boxplot(test$ApplicantIncome,test$CoapplicantIncome,names = c("App income","coapp income"),main="test")

par(mfrow=c(1,2))
hist(train$Loan_Amount_Term,breaks=500,main="train")
hist(test$Loan_Amount_Term,breaks=500,main="test")
#property area
par(mfrow=c(1,2))
barplot(table(train$Property_Area),main="train")
barplot(table(test$Property_Area),main="test")

#Loan Status by Gender of Applicant
print(ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Gender)+ggtitle("Loan Status by Gender of Applicant"))
print(ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Dependents)+ggtitle("Loan Status by number of Dependents of Applicant"))

#this looks very important! Almost all applicants with history=0 are refused
print(ggplot(train, aes(x=Loan_Status))+geom_bar()+facet_grid(.~Credit_History)+ggtitle("Loan Status by credit history of Applicant"))
print(ggplot(data=train[train$ApplicantIncome<20000,],aes(ApplicantIncome,fill=Married))+geom_bar(position="dodge")+facet_grid(Gender~.))

#add some feature engineering
alldata<-rbind(train[,2:12],test[,2:12])
depFit <- rpart(data=alldata,Dependents~.,xval=3)
fancyRpartPlot(depFit)
alldata$numDependents <- recode(alldata$Dependents,"'3+'='3' ")
alldata$FamilySize <- ifelse((alldata$CoapplicantIncome>0 |alldata$Married=="Y"),numDependents+2,numDependents+1)
alldata$TotalIncome<-alldata$ApplicantIncome +alldata$CoapplicantIncome
alldata$IncomePC <- alldata$TotalIncome/alldata$FamilySize
alldata$LoanAmountByTotInc <- alldata$LoanAmount/alldata$TotalIncome
alldata$LoanAmountPC <- alldata$LoanAmount/alldata$IncomePC
alldata$LoanPerMonth <- alldata$LoanAmount/alldata$Loan_Amount_Term
alldata$LoanPerMOnthByTotInc  <- alldata$LoanPerMonth/alldata$TotalIncome
alldata$LoanPerMonthPC <- alldata$LoanPerMonth/alldata$LoanAmountPC
alldata$Loan_Amount_Term <- as.factor(alldata$Loan_Amount_Term)



#log transformations for monetary variables
bins<-cut(alldata$ApplicantIncome,breaks=20)
barplot(table(bins),main="Applicant Income")
logbins<-cut(ifelse(alldata$ApplicantIncome<2.72,0,log(alldata$ApplicantIncome)),breaks=20)
barplot(table(logbins),main="Log of Applicant Income")

alldata$LogApplicantIncome <- ifelse(alldata$ApplicantIncome<2.72,0,log(alldata$ApplicantIncome))
alldata$LogCoapplicantIncome <- ifelse(alldata$CoapplicantIncome<2.72,0,log(alldata$CoapplicantIncome))
alldata$LogLoanAmount <- log(alldata$LoanAmount)
alldata$LogTotalIncome <- log(alldata$TotalIncome)
alldata$IncomePC <- log(alldata$IncomePC)
alldata$LogLoanAmountPC <- log(1000*alldata$LoanAmountPC)
alldata$LogLoanPerMOnth <- log(alldata$LoanPerMonth)
alldata$LogLoanPerMOnthPC <- log(alldata$LoanPerMonthPC)

#remove variables that are highly correlated

newtrain <- alldata[1:614,]
newtest <- alldata[615:981,]
newtrain$Loan_Status<-train$Loan_Status

#random forest predictive analytics
newtrain$Dependents<-as.factor(newtrain$Dependents)
fit <- randomForest(as.factor(Loan_Status) ~ .,data=newtrain, importance=TRUE, ntree=2000)
varImpPlot(fit)


newtest$Dependents<-as.factor(newtest$Dependents)
Prediction <- predict(fit, newtest)
submit<-data.frame(Loan_ID=test$Loan_ID,Loan_Status=Prediction)
write.csv(submit,file="randomforest_sol_AFTER SOME_featured_engineering.csv")

#condition based random forest
set.seed(415)
fit <- cforest(as.factor(Loan_Status) ~ .,data = newtrain, controls=cforest_unbiased(ntree=2000, mtry=3)) 
Prediction <- predict(fit, newtest, OOB=TRUE, type = "response")
submit<-data.frame(Loan_ID=test$Loan_ID,Loan_Status=Prediction)
write.csv(submit,file="condition_basedrandomforest_sol_AFTER SOME_featured_engineering.csv")
#confusion matrix
#Confusion Matrix and Statistics

#Reference
#Prediction   N   Y
#N 192   0
#Y   0 422

#Accuracy : 1         
#95% CI : (0.994, 1)
#No Information Rate : 0.6873    
#P-Value [Acc > NIR] : < 2.2e-16 

#Kappa : 1         
#Mcnemar's Test P-Value : NA        

#Sensitivity : 1.0000    
#Specificity : 1.0000    
#Pos Pred Value : 1.0000    
#Neg Pred Value : 1.0000    
#Prevalence : 0.3127    
#Detection Rate : 0.3127    
#Detection Prevalence : 0.3127    
#Balanced Accuracy : 1.0000    

#'Positive' Class : N   

#Tuning the ramdom  forest model and decision tree
#create task
trainTask <- makeClassifTask(data = newTrain,target = "Loan_Status")
testTask <- makeClassifTask(data = newTest, target = "Loan_Status")

#normalize the variables
trainTask <- normalizeFeatures(trainTask,method = "standardize")
testTask <- normalizeFeatures(testTask,method = "standardize")

tree <- makeLearner("classif.rpart", predict.type = "response")
#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#Search for hyperparameters
treepars <- makeParamSet(
  makeIntegerParam("minsplit",lower = 10, upper = 50),
  makeIntegerParam("minbucket", lower = 5, upper = 50),
  makeNumericParam("cp", lower = 0.001, upper = 0.2)
)
#try 100 different combinations of values
tpcontrol <- makeTuneControlRandom(maxit = 100L)

#hypertune the parameters
set.seed(11)
treetune <- tuneParams(learner = tree, resampling = set_cv, task = trainTask, par.set = treepars, control = tpcontrol, measures = acc)
treetune
#using hyperparameters for modeling
tunedtree <- setHyperPars(tree, par.vals=treetune$x)

#train the model
treefit <- train(tunedtree, trainTask)
par(mfrow=c(1,1))
fancyRpartPlot(getLearnerModel(treefit))
#make predictions
treepred <- predict(treefit, testTask)

#create a submission file
submit1 <- data.frame(Loan_ID = test$Loan_ID, Loan_Status = treepred$data$response)

#create a learner
rf <- makeLearner("classif.randomForest", predict.type = "response"
                  , par.vals = list(ntree = 200, mtry = 3))
rf$par.vals <- list(importance = TRUE)

#set tunable parameters
rf_param <- makeParamSet(
  makeIntegerParam("ntree",lower = 50, upper = 500),
  makeIntegerParam("mtry", lower = 2, upper = 10),
  makeIntegerParam("nodesize", lower = 10, upper = 50)
)

#let's do random search for 100 iterations
rancontrol <- makeTuneControlRandom(maxit = 100L)

#set 3 fold cross validation
set_cv <- makeResampleDesc("CV",iters = 3L)

#hypertuning
set.seed(11)
rf_tune <- tuneParams(learner = rf, resampling = set_cv, task = trainTask, par.set = rf_param, control = rancontrol, measures = acc)
#cv accuracy
rf_tune$y

#using hyperparameters for modeling
tunedrf <- setHyperPars(rf, par.vals = rf_tune$x)

#train a model
rforest <- train(tunedrf, trainTask)
getLearnerModel(rforest)
#Call:
#  randomForest(formula = f, data = data, classwt = classwt, cutoff = cutoff,      importance = TRUE, ntree = 161L, mtry = 9L, nodesize = 49L) 
#Type of random forest: classification
#Number of trees: 161
#No. of variables tried at each split: 9

#OOB estimate of  error rate: 18.57%
#Confusion matrix:
#  N   Y class.error
#N 96  96  0.50000000
#Y 18 404  0.04265403
