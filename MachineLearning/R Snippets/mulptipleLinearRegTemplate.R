#Data Preprocessing and mdoel training with Caret

#Importing the dataset
setwd('D:\\Downloads\\MachineLearningDownloads\\Multiple_Linear_Regression')
dataset = read.csv('50_Startups.csv', stringsAsFactors = T)

#Preprocessing
library(caret)

sum(is.na(dataset))

#Imputing possible missing values, scaling and centering
preProcValues<- preProcess(dataset, method = c("knnImpute","center","scale"))

library(RANN)
train_processed<- predict(preProcValues, dataset)

sum(is.na(dataset))

#Encode cat vars
dmy<- dummyVars("~.", data = dataset, fullRank = T)
#fullRank is used to avoid dummy trap
dataset<- data.frame(predict(dmy,newdata = dataset))


#Splitting the data
index<- createDataPartition(dataset$Profit, p = 0.8, list = F)
train<- dataset[index,]
test<- dataset[-index,]

#Recursive Feature Selection
#rfe uses rasampling while rfeIter des not
#The rfeControl param is needed to define the mode
#and the methods used to create the models that will
#be compared
control<- rfeControl(functions = lmFuncs, #Linear models
                     method = "repeatedcv",
                     repeats = 5,
                     verbose = F)
#In R we start counting from 1 not 0
profitProfile<- rfe(train[, -6], train[,'Profit'],
                rfeControl = control)
profitProfile

predictors<- predictors(profitProfile)
profitProfile$fit #Fits the model with the best subset

#Plot the model
trellis.par.set(caretTheme())
plot(profitProfile)

#Train the model
#Caret has 200+ ML algos
names(getModelInfo())

#We used linear regression with stepwise selection and AIC
#Probably the step above is not necessary at least in lm
model<- train(train[,1:5], train[,6], method = 'lmStepAIC')
summary(model)

#Parameter tuning
fitCOntrol<- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5
)

#Create the grid
grid<- expand.grid(n.trees = c(10,20,50,100,500,1000),
                   shrinkage = c(0.01,0.05,0.1,0.5),
                   n.minobinnode = c(3,5,10),
                   interaction.depth = c(1,5,10))

#Retrain the model
model<- train(train[,1:5], train[,6], method = 'lmStepAIC',
              trControl=fitCOntrol, tuneGrid=grid)

#Predictions
predictions<- predict.train(object = model, test[,1:5], type = 'raw')
confusionMatrix(predictions, test[,6])
