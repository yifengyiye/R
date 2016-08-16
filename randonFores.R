install.packages("randomForest")
library(randomForest)
data <- iris
data

Randommodel <- randomForest(Species ~ ., data=data, importance=TRUE,proximity=FALSE,ntree=100)
Randommodel
system.time(Randommodel)
print(Randommodel)

importance(Randommodel, type=1)
importance(Randommodel, type=2)
varImpPlot(Randommodel)

# 模型预测能力：
predict(object, newdata, type="response", norm.votes=TRUE,
        predict.all=FALSE,proximity=FALSE,nodes=FALSE,outoff,...)
# nodes判断是否是终点，proximity判断是否需要进行近邻测量，prodict.all判断是狗保留所有的预测值
prediction = predict(Randommodel, data[,1:5], type="class")
table(observed=data$Species,predicted=prediction)

#随机森林包
install.packages("party")
library(party)

set.seed(42)
crf <- cforest(y~.,control=cforest_unbiased(mtry=2,ntree=50),data=step2_1)
varimpt <- data.frame(varimp(crf))

