#load packages
library('caret')
library('ggplot2')
library('ggthemes')
names(getModelInfo())
library('data.table')
library('gridExtra')
library('corrplot')
library('nnet')
library('pROC')
#set the working directory
setwd("~/MIT Big Data Science/MIT 801 - Introduction to Machine Learning and Statistical Learning/Assignment 2")

#Import dataset
covtype1<-gzfile('covtype.data.gz')   
dataset<-read.table(covtype1,header=F,sep = ",")

#Rename columns
setnames(dataset, old = c('V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13',
                           'V14','V15','V16','V17','V18','V19','V20','V21','V22','V23','V24','V25','V26',
                           'V27','V28','V29','V30','V31','V32','V33','V34','V35','V36','V37','V38','V39',
                           'V40','V41','V42','V43','V44','V45','V46','V47','V48','V49','V50','V51','V52',
                           'V53','V54','V55'), 
                new =c('Elevation','Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
               'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points','Wilderness_Area1','Wilderness_Area2','Wilderness_Area3',
               'Wilderness_Area4','Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10'
               ,'Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21'
               ,'Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32'
               ,'Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Cover_Type'))
#The dataset contains 581 012 observations, 54 independent variables consisting of 40 soil types, 4 areas of wilderness, Elevation, Aspect, Slope,
#Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points
# It also has a Cover_Type variable that is a class variables with 7 cover types

#Checking for missing values
#No missing values
sapply(dataset, function(x) sum(is.na(x)))

soil<- dataset[ ,c(15:54)]
area<- dataset[,c(11:14)]
dataset<- dataset[,c(-15:-54, -11:-14)]
Newfactor <- factor(apply(soil, 1, function(x) which(x == 1)), labels = c(1:40)) 
dataset$Soil_Type<- as.integer(Newfactor)
Newfactor2 <- factor(apply(area, 1, function(x) which(x == 1)), labels = c(1:4)) 
dataset$Wilderness_Area<- as.integer(Newfactor2)
dataset<- dataset[ ,c(1:10,12,13,11)]

boxplot(dataset[,c(-7,-8,-9,-11,-12,-13)], las=3, par(mar = c(15, 4, 2, 2)), col="blue",main="Boxplot for some attributes")
theme_set(theme_grey(base_size = 20))

g1<- ggplot(dataset, aes(Elevation, color = factor(Cover_Type), fill = factor(Cover_Type))) + geom_density(alpha = 0.2)
g2<- ggplot(dataset, aes(Aspect, color = factor(Cover_Type), fill = factor(Cover_Type))) + geom_density(alpha = 0.2)
g3<- ggplot(dataset, aes(Horizontal_Distance_To_Roadways, color = factor(Cover_Type), fill = factor(Cover_Type))) + geom_density(alpha = 0.2)
g4<- ggplot(dataset, aes(Horizontal_Distance_To_Fire_Points, color = factor(Cover_Type), fill = factor(Cover_Type))) + geom_density(alpha = 0.2)
grid.arrange(g1, g2,g3,g4, ncol=2,nrow=2)

#Correlation matrix
cor<- dataset[,c(-9,-8,-7,-13)]
names(cor)<- c("Elevation", "Aspect","Slope","H_D_To_Hydro","V_D_To_Hydro","H_D_To_Roads", "H_D_To_Fire_Points" ,"Soil_Type","Wilderness_Area" )
#Correlation between variables
m<- cor(cor)
corrplot(m, method = "number") 
#tl.cex=1.2) 

#Plotting Cover_Type vs some features 
ggplot(dataset, aes(x = Cover_Type, y=Elevation)) +
    geom_point(aes(colour = factor(Soil_Type),size = 3)) + scale_x_discrete(limits=c(1,2,3,4,5,6,7)) +ggtitle(label = "Elevation vs Cover_Type")+theme_few()

ggplot(dataset, aes(x = Cover_Type, y=Soil_Type)) +
    geom_point(aes(colour = factor(Soil_Type))) + 
   scale_x_discrete(limits=c(1,2,3,4,5,6,7)) +ggtitle(label = "Soil_Type vs Cover_Type")+theme_few()

ggplot(dataset, aes(x = Cover_Type, y=Wilderness_Area)) +
    geom_point(aes(colour = factor(Soil_Type),size=3)) + scale_x_discrete(limits=c(1,2,3,4,5,6,7)) +ggtitle(label = "Area of Wilderness vs Cover_Type")+theme_few()



#Split dataset into train and test set
dataset$Cover_Type<-as.factor(dataset$Cover_Type)

splitIndex<-createDataPartition(dataset$Cover_Type,p=0.2, list=FALSE,times = 1)
trainCT<-dataset[splitIndex,]
testCT<-dataset[-splitIndex,]

maxs <- apply(trainCT[-13], 2, max)
mins <- apply(trainCT[-13], 2, min)
trainCT[-13]<-scale(trainCT[-13], center = mins, scale = maxs - mins)

maxs <- apply(testCT[-13], 2, max)
mins <- apply(testCT[-13], 2, min)
testCT[-13]<-scale(testCT[-13], center = mins, scale = maxs - mins)
#create dummy variables
#Y<-class2ind(factor(dataset$Cover_Type))

#trainCT[] <- lapply(trainCT, factor)

#Modeling

#KNN
library(class)
y_pred = knn(train = trainCT[, -13],
             test = testCT[, -13],
             cl = trainCT[, 13],
             k = 5,
             prob = TRUE)
cm = confusionMatrix(testCT[, 13], y_pred)


#RandomForest
install.packages('randomForest')
library(randomForest)
set.seed(123)
RF_model = randomForest(x = trainCT[-13],
                          y = trainCT$Cover_Type,
                          ntree = 10)

# Predicting the Test set results
y_pred1 = predict(RF_model, newdata = testCT[,-13])

# Making the Confusion Matrix
cm1 = confusionMatrix(testCT[, 13], y_pred1)
varImp(RF_model,col="darkblue", pch=19)


#Decision Tree
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
set.seed(123)
DT_model = rpart(formula = Cover_Type ~.,
                   data = trainCT,method="class")

# Predicting the Test set results
y_pred2 = predict(DT_model, newdata = testCT[,-13], type = 'class')

# Making the Confusion Matrix
cm2 = confusionMatrix(testCT[, 13], y_pred2)

# Draw the complex tree
fancyRpartPlot(DT_model)

# Prune the tree: pruned
pruned <- prune(DT_model, cp = 0.01) 

# Draw pruned
fancyRpartPlot(pruned)
library(ROCR)
pred1 <- prediction(predict(DT_model), trainCT$Cover_Type)
perf1 <- performance(pred1,"tpr","fpr")
plot(perf1)

#SVM
install.packages('e1071')
library(e1071)
SVM_model = svm(formula = Cover_Type ~ .,
                 data = trainCT,
                 type = 'C-classification',
                 kernel = 'radial')

# Predicting the Test set results
y_pred3 = predict(SVM_model, newdata = testCT[,-13])

# Making the Confusion Matrix
cm3 = confus(testCT[,13], y_pred3)

#NNET
library(nnet)
install.packages('nnet',dependencies=TRUE)
idC <-class2ind(trainCT$Cover_Type)
NN1=nnet(trainCT, idC[trainCT], size=10, maxit = 200, softmax=TRUE)
predict(NN1, data=testCT,type = "class")

# Predicting the Test set results
y_pred4 = predict(NN1_model, newdata = testCT[,-13])



































































































#cv - cross validation done twice
fitControl<-trainControl(method='cv',number = 5)

#preProcValues <- preProcess(x = trainX,method = c("center", "scale"))




#Modeling

#K-Nearest neighbour
set.seed(7)
KNN_model<-train(Cover_Type~.,data=trainCT, method="knn",trControl=fitControl)
plot(KNN_model$finalModel)
predictions3 <- predict(KNN_model, testCT[,-13])
# summarize results
confusionMatrix(predictions3, testCT[,13])
print(KNN_model)
print(KNN_model$bestTune)
print(KNN_model$results)

#Decision Tree
set.seed(7)
DecisionTree_model<-train(Cover_Type~Elevation+Wilderness_Area+Soil_Type,data=trainCT,method="rpart", trControl=fitControl)
plot(DecisionTree_model$finalModel)
predictions2 <- predict(DecisionTree_model, testCT[,-13])
# summarize results
confusionMatrix(predictions2, testCT[,13])
print(DecisionTree_model)
print(DecisionTree_model$bestTune)
print(DecisionTree_model$results)

#Decision (Random) Forest
set.seed(7)
RF_model<-train(Cover_Type~.,data=trainCT,method="rf",ntree=10 ,trControl=fitControl,tunelength=5)
plot(RF_model$finalModel)
predictions <- predict(RF_model, testCT[,-13])
# summarize results
confusionMatrix(predictions, testCT[,13])
print(RF_model)
print(RF_model$bestTune)
print(RF_model$results)


#Neural Network
set.seed(7)
NN_model<-train(Cover_Type~.,data=trainCT,method="nnet", trControl=fitControl)
plot(NN_model$finalModel)
predictions1 <- predict(NN_model, testCT[,-13])
# summarize results
confusionMatrix(predictions1, testCT[,13])
print(NN_model)
print(NN_model$bestTune)
print(NN_model$results)


#SVM
set.seed(7)
SVMRadial_model<-train(Cover_Type~.,data=trainCT, method="svmLinear", trControl=fitControl,tunelength=5)
plot(SVMRadial_model$finalModel)
predictions4 <- predict(SVMRadial_model, testCT[1:300000,-13])
# summarize results
confusionMatrix(predictions4, testCT[1:300000,13])
print(SVMRadial_model)
print(SVMRadial_model$bestTune)
print(SVMRadial_model$results)



# collect resamples
results <- resamples(list(KNN=KNN_model, DecisionTree=DecisionTree_model, 
                          RandomForest=RF_model,NeuralNetwork=NN_model,SVM=SVMRadial_model))

# summarize the distributions
summary(results)

# boxplots of results
bwplot(results)
dotplot(results)

#Predictions on test set

### KNN Model Predictions and Performance
# Make predictions using the test data set
KNN.pred <- predict(KNN_model,testCT[,-13])

#Look at the confusion matrix  
confusionMatrix(KNN.pred,testCT$Cover_Type)   

#Draw the ROC curve 
KNN.probs <- predict(KNN_model,testCT[,-13],type="prob")
head(KNN.probs)

KNN.ROC <- roc(predictor=KNN.probs$PS,
               response=testCT$Cover_Type,
               levels=rev(levels(testCT$Cover_Type)))
KNN.ROC$auc
#Area under the curve: 
plot(KNN.ROC,main="KNN ROC")



### Decision Tree Model Predictions and Performance
# Make predictions using the test data set
DT.pred <- predict(DecisionTree_model,testCT[,-13])

#Look at the confusion matrix  
confusionMatrix(DT.pred,testCT$Cover_Type)   

#Draw the ROC curve 
DT.probs <- predict(DecisionTree_model,testCT[,-13],type="prob")
head(DT.probs)

DT.ROC <- roc(predictor=DT.probs$PS,
              response=testCT$Cover_Type,
              levels=rev(levels(testCT$Cover_Type)))
DT.ROC$auc
#Area under the curve: 
plot(DT.ROC,main="DT ROC")


### Random Forest Model Predictions and Performance
# Make predictions using the test data set
RF.pred <- predict(RF_model,testCT[,-13])

#Look at the confusion matrix  
confusionMatrix(RF.pred,testCT$Cover_Type)   

#Draw the ROC curve 
RF.probs <- predict(RF_model,testCT[,-13],type="prob")
head(RF.probs)

selectedIndices <- RF_model$pred$mtry == 2
# Plot:

RF.ROC$auc
#Area under the curve: 
plot(RF.ROC,main="RF ROC")


### neural Network Model Predictions and Performance
# Make predictions using the test data set
NN.pred <- predict(NN_model,testCT[,-13])

#Look at the confusion matrix  
confusionMatrix(NN.pred,testCT$Cover_Type)   

#Draw the ROC curve 
NN.probs <- predict(NN_model,testCT[,-13],type="prob")
head(NN.probs)

NN.ROC <- roc(predictor=NN.probs$PS,
              response=testCT$Cover_Type,
              levels=rev(levels(testCT$Cover_Type)))
NN.ROC$auc
#Area under the curve: 
plot(NN.ROC,main="NN ROC")


### Random Forest Model Predictions and Performance
# Make predictions using the test data set
SVM.pred <- predict(SVM_model,testCT[,-13])

#Look at the confusion matrix  
confusionMatrix(SVM.pred,testCT$Cover_Type)   

#Draw the ROC curve 
SVM.probs <- predict(SVM_model,testCT[,-13],type="prob")
head(SVM.probs)

SVM.ROC <- roc(predictor=SVM.probs$PS,
               response=testCT$Cover_Type,
               levels=rev(levels(testCT$Cover_Type)))
SVM.ROC$auc
#Area under the curve: 
plot(SVM.ROC,main="RF ROC")
