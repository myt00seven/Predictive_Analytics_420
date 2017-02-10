setwd("/Users/kungangzhang/Documents/OneDrive/Northwestern/Study/Courses/MSiA-420-0/HW2")
rm(list = ls())
require(gdata)
set.seed(111)

#Prob 1)
##(a)Fit a linear model and discuss the predictive power.
##Answer: First I take log transform to the cost, the response variable, and then fit the model with all predictors unchanged. The $R^2$ is $0.5831$. Then, I tried to standardize everything and the $R^2$ is $0.5527$. I saw some of predictors also have skewed distribution or long tail problem, so that I try log transform (or some special log transform depending on whether it is left-skewed and right-skewed), and the histograms look more symmetric. For the rest of predictors, I just let them be. The $R^2$ increases to $0.658$. Generally, those predictors significant before transform are also significant afterwards. 
##The histogram of each columns
df0<-read.xls("./HW2_data.xls",sheet=1,header=TRUE)
par(mfrow=c(3,3))
df<-df0
df$gend <- as.factor(df$gend)
par(mfrow=c(3,3))
for (i in seq(2,10)) hist(df0[[i]],breaks=30,xlab=names(df0)[i])
df$cost <- log10(df$cost)
mod1<-lm(cost~.,data = df[-1])
summary(mod1)
##Also, I tried to standardize each variable to see effect.
df_std<-df
df_std[c(2,3,5:10)]<-sapply(df_std[c(2,3,5:10)], function(x) (x-mean(x))/sd(x))
mod2<-lm(cost~.,data = df_std[-1])
summary(mod2)
##Except the intercept, the other aspects of the linear model would not be changed by standardization, given the full-rank design matrix. 

##As shown in the histogram plots the columns of cost, num of interventions, num of drugs, num of emergency, num of complication, num of other diseaes are left-skewed, while the age is right-skewed. So I need to do a log transform to those columns left-skewed and do a special log transform to those columns right-skewed.
par(mfrow=c(3,3))
for (i in seq(2,10)) {
  if (i %in% c(2,5,6,7,9)){
    hist(log10(df0[[i]]+1),breaks=30,xlab=paste(names(df0)[i],'log',sep='-'))  
  }
  if (i==3){
    hist(log10(max(df0[[i]])+1-df0[[i]]),breaks=30,xlab = paste(names(df0)[i],'log','special',sep='-'))
  }
  if (i %in% c(4,8,10)){
    hist(df0[[i]],breaks=30,xlab=names(df0)[i])
  }
  
}
df_trans_std <- df
df_trans_std$age <- log10(max(df_trans_std$age)+1-df_trans_std$age)
df_trans_std$intvn <- log10(df_trans_std$intvn+1)
df_trans_std$drugs <- log10(df_trans_std$drugs+1)
df_trans_std$ervis <- log10(df_trans_std$ervis+1)
df_trans_std$comorb <- log10(df_trans_std$comorb+1)
df_trans_std[c(2,3,5:10)]<-sapply(df_trans_std[c(2,3,5:10)], function(x) (x-mean(x))/sd(x))
mod3<-lm(cost~.,data = df_trans_std[-1])#no matter use log10(age+1) or log10(I(age+1)), the result is the same.
summary(mod3)
##After the log transform, we see the $R^2$ increases and the influential predictors don't change much.

##(From website by searching 'long tail distribution log transform') You don't need to assume a lognormal distribution; there's no requirement that an independent variable in linear regression itself has a normal distribution. The hope is that, with log transformation of the independent variable, the other requirements for interpreting linear regression results will better be met, such as having normally distributed residual errors independent of fitted values. If the regression against the log-transformed independent variable meets those requirements, there are no problems with interpreting p-values, etc. Regression coefficients will now mean the change in the dependent variable per log change in the independent variable. So if you use log10, the regression coefficient will be "change per 10-fold change in GDP" for your example; for log2, "change per doubling of GDP."

##(b)Which variables appear to have the most influence on the cost.
##Answer: From the mod2 (the standardized model without log tranforming predictors), we have the biggest coefficient of number of interventions ($0.59335$), so that this predictor would have the most influence on the cost. Similarily, in the mod3 (standardized model with log tranforming predictors) the number of intervention also has the biggest influence ($0.642449$).

##(c)Construct appropriate diagnostics and residual plots to assess (related to nonlinearity in the relation b/w the response and the predictors.)
##Answer: From both of the plots below, we saw the residuals have little correlation with predictors, so that we don't need to change the model. If there were any nonlinear correlation, we probably need to design better predictors to capture this nonlinearity.
##For linear model without log transforming predictors
par(mfrow=c(2,4),pin=c(0.8,0.8),tcl=-0.15,mgp=c(1,0.2,0))
for (i in seq(3:10)) {
  plot(df_std[[i+2]],resid(mod2),ylab="Residuals",xlab=names(df_std)[i+2],main="")
  abline(0, 0)}
title(main="Ischemic heart disease-standardized predictors with log(cost)-lm",outer = T)
##For linear model with log transforming skewed predictors
par(mfrow=c(2,4),pin=c(0.8,0.8),tcl=-0.15,mgp=c(1,0.2,0))
for (i in seq(3:10)) {
  plot(df_trans_std[[i+2]],resid(mod2),ylab="Residuals",xlab=names(df_trans_std)[i+2],main="")
  abline(0, 0)}
title(main="Ischemic heart disease-standardized predictors with log(cost)-lm-Log transforming predictors",outer = T)

#Prob 2)Find the best neural network model for the ischemic heart disease data set, using linear output activation func, and do not rescale the response.
##(a)Use 10-fold CV to find the best combination of shrinkage param and the number of hidden nodes.
##Answer:

##CV index random generator
CVInd <- function(n,K) {  #n is sample size; K is number of parts; returns K-length list of indices for each part
  m<-floor(n/K)  #approximate size of each part
  r<-n-m*K  
  I<-sample(n,n)  #random reordering of the indices
  Ind<-list()  #will be list of indices for all K parts
  length(Ind)<-K
  for (k in 1:K) {
    if (k <= r) kpart <- ((m+1)*(k-1)+1):((m+1)*k)  
    else kpart<-((m+1)*r+m*(k-r-1)+1):((m+1)*r+m*(k-r))
    Ind[[k]] <- I[kpart]  #indices for kth part of data
  }
  Ind
}

##Now use multiple reps of CV to compare Neural Nets and linear reg models###
library(nnet)
Nrep<-10 #number of replicates of CV
K<-10  #K-fold CV on each replicate
n.lam = 10 #number of lambda
n.num_hidnode = 3 #number of different numbers of hidden nodes
n.models = n.lam*n.num_hidnode #number of different models to fit
n=nrow(df_std)
y<-df_std$cost
yhat=matrix(0,n,n.models)
lam_seq = 10^seq(-as.integer(n.lam/2),as.integer(n.lam/2)-1)
num_hidnode_seq = 5*seq(1,n.num_hidnode) 
mod_par=matrix(c(rep(lam_seq,times=1,each=n.num_hidnode),rep(num_hidnode_seq,times=n.lam,each=1)),2,n.models,byrow = T)#Store the model parameters: lambda and the number of nodes in hidden layer
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  print(c(0,0,0,j))#Print out the index of replicates of CV
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    print(k)#Print out the index of different fold of CV
    for (m in 1:n.models) {
      out<-nnet(cost~.,df_std[-Ind[[k]],],linout = T, skip=F,size=as.integer(mod_par[2,m]),decay=mod_par[1,m],maxit=1000,trace=F)
      yhat[Ind[[k]],m]<-as.numeric(predict(out,df_std[Ind[[k]],]))
    }
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSE
MSEAve<- apply(MSE,2,mean); MSEAve #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); MSEsd   #SD of mean square CV error
r2<-1-MSEAve/var(y); r2  #CV r^2
##The best model in terms of the minimum MSEAve or the maximum r2.
min(MSEAve)
max(r2)
##Return the index of the minimum MSEAve or the maximum r2.
which(MSEAve==min(MSEAve))
which(r2==max(r2))
##The optimal lambda and number of hidden nodes
mod_par[,which(MSEAve==min(MSEAve))]

##Do a CV in smaller interval of lambda and number of hidden nodes again.
Nrep<-10 #number of replicates of CV
K<-10  #K-fold CV on each replicate
n.lam = 19 #number of lambda
n.num_hidnode = 5 #number of different numbers of hidden nodes
n.models = n.lam*n.num_hidnode #number of different models to fit
n=nrow(df_std)
y<-df_std$cost
yhat=matrix(0,n,n.models)
lam_seq = c(seq(1,9),seq(10,100,10))
num_hidnode_seq = seq(12,20,2) 
mod_par=matrix(c(rep(lam_seq,times=1,each=n.num_hidnode),rep(num_hidnode_seq,times=n.lam,each=1)),2,n.models,byrow = T)#Store the model parameters: lambda and the number of nodes in hidden layer
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  print(c(0,0,0,j))#Print out the index of replicates of CV
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    print(k)#Print out the index of different fold of CV
    for (m in 1:n.models) {
      out<-nnet(cost~.,df_std[-Ind[[k]],],linout = T, skip=F,size=as.integer(mod_par[2,m]),decay=mod_par[1,m],maxit=1000,trace=F)
      yhat[Ind[[k]],m]<-as.numeric(predict(out,df_std[Ind[[k]],]))
    }
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum((y-x)^2))/n
} #end of j loop
MSE
MSEAve<- apply(MSE,2,mean); MSEAve #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); MSEsd   #SD of mean square CV error
r2<-1-MSEAve/var(y); r2  #CV r^2
##The best model in terms of the minimum MSEAve or the maximum r2.
min(MSEAve)
max(r2)
##Return the index of the minimum MSEAve or the maximum r2.
which(MSEAve==min(MSEAve))
which(r2==max(r2))
##The optimal lambda and number of hidden nodes
par_best<-mod_par[,which(MSEAve==min(MSEAve))]

##(b)Fit the best model and discuss how good the predictive power is.
##Answer: The cross-validation $R^2$ of the best model is , with the penalization and number of hidden nodes as .
nnet_mod<-nnet(cost~.,df_std,linout = T, skip=F,size=as.integer(par_best[2]),decay=par_best[1],maxit=1000,trace=F)
summary(nnet_mod)

##(c)The variables having the most influence on cost (Use the ALEPlot package for this).
##Answer: 
library(ALEPlot)
yhat <- function(X.model, newdata) as.numeric(predict(X.model, newdata))
par(mfrow=c(2,4),pin=c(0.7,0.7),tcl=-0.2,mgp = c(1,0.15,0))
for (j in 3:10)  {ALEPlot(df_std, nnet_mod, pred.fun=yhat, J=j, K=50, NA.plot = TRUE)
  rug(df_std[,j]) }  ## This creates main effect ALE plots for all 8 predictors
par(mfrow=c(1,1))

par(mfrow=c(2,2),pin=c(1.3,1.3),mgp = c(1,0.15,0),tcl=-0.15)  
## This creates 2nd-order interaction ALE plots for x1, x2, x8
ALEPlot(df_std, nnet_mod, pred.fun=yhat, J=c(3,7), K=50, NA.plot = TRUE)
ALEPlot(df_std, nnet_mod, pred.fun=yhat, J=c(3,6), K=50, NA.plot = TRUE)
ALEPlot(df_std, nnet_mod, pred.fun=yhat, J=c(7,8), K=50, NA.plot = TRUE)
ALEPlot(df_std, nnet_mod, pred.fun=yhat, J=c(5,10), K=50, NA.plot = TRUE)

##(d)Construct appropriate residual plots to assess the nonlinearity not captured by the nnet.
par(mfrow=c(1,1),pin=c(4,4))
plot(df$cost,resid(nnet_mod),ylab="Residuals", xlab="cost", main="Ischemic heart disease-standardized \n predictors with log(cost)-nnet")
abline(0, 0)

#Prob 3)Repeat Prob 2) but for a regression tree.

##(a)Use 10-fold CV to find the best tree size or complexity parameter value
#do not have to standardize or transform predictors to fit trees
# the CV shell is not correct in tree?
# cp is \lambda, the complex parameter; with small cp we will grow a big tree(overfit)
# xval: fold of cross validation
library(rpart)
control <- rpart.control(minbucket = 5, cp = 0.0001, maxsurrogate = 0, usesurrogate = 0, xval = 10)
par(mfrow=c(1,1),pin=c(4,4),mgp=c(2,1,0))
df_std.tr <- rpart(cost ~ .,df_std, method = "anova", control = control)
plotcp(df_std.tr)  #plot of CV r^2 vs. size
printcp(df_std.tr)  #same info is in df_std.tr$cptable
#prune back to optimal size, according to plot of CV 1-r^2
df_std.tr1 <- prune(df_std.tr, cp=0.00631770)  #approximately the best size pruned tree
df_std.tr1$variable.importance#The importance of each predictors
df_std.tr1$cptable[nrow(df_std.tr1$cptable),] #shows training and CV 1-r^2, and other things
#prune and plot a little smaller tree than the optimal one, just for display
df_std.tr2 <- prune(df_std.tr, cp=0.00631770)  #bigger cp gives smaller size tree
df_std.tr2
par(cex=.5); plot(df_std.tr2, uniform=F); text(df_std.tr2, use.n = T); par(cex=1)
##
yhat<-predict(df_std.tr1); e<-df_std$cost-yhat
c(1-var(e)/var(df_std$cost), 1-df_std.tr1$cptable[nrow(df_std.tr1$cptable),3]) #check to see training r^2 agrees with what is in cptable

##(b)Fit the best model and discuss how good the predictive power of model is.
control_best <- rpart.control(minbucket = 5, cp = 0.00631770, maxsurrogate = 0, usesurrogate = 0)
df_std.tr_best <- rpart(cost ~ .,df_std, method = "anova", control = control)
summary(df_std.tr_best)

##(c)The most influencing variable on the cost and the effect.
##Answer: From the above result, the intvn has the most influence on the cost, and the effect is the larger intvn, the more of final cost.

##(d)Construct appropriate residual plots to assess whether there remains any linearity not captured by the regression tree model.
##Answer:
par(mfrow=c(1,1),pin=c(4,4))
plot(df$cost,resid(df_std.tr2),ylab="Residuals", xlab="cost", main="Ischemic heart disease-standardized \n predictors with log(cost)-reg tree")
abline(0, 0)

##(e)Linear reg, nnet, tree, which you recommand for this data set and why?

#Prob 4)Forensic example, keep all 6-category to do classification
FGL<-read.table("../Data_for_Lecture_Examples/fgl.txt",sep="\t")
FGL1<-FGL
k<-ncol(FGL)-1;
FGL1[1:k]<-sapply(FGL1[1:k], function(x) (x-mean(x))/sd(x))
FGL1<-data.frame(FGL1,"type_ind"=as.numeric(factor(FGL1$type)))
##Or use: as.numeric(factor(FGL1$type, levels=levels(FGL1$type)))

##(a)10-fold CV to find the best nnet for classifying the class type
library(nnet)
Nrep<-10 #number of replicates of CV
K<-10  #K-fold CV on each replicate
n.lam = 10 #number of lambda
n.num_hidnode = 3 #number of different numbers of hidden nodes
n.models = n.lam*n.num_hidnode #number of different models to fit
n=nrow(FGL1)
y<-FGL1$type_ind
yhat=matrix(0,n,n.models)
lam_seq = 10^seq(-as.integer(n.lam/2),as.integer(n.lam/2)-1)
num_hidnode_seq = 5*seq(1,n.num_hidnode) 
mod_par=matrix(c(rep(lam_seq,times=1,each=n.num_hidnode),rep(num_hidnode_seq,times=n.lam,each=1)),2,n.models,byrow = T)#Store the model parameters: lambda and the number of nodes in hidden layer
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  print(c(0,0,0,j))#Print out the index of replicates of CV
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    print(k)#Print out the index of different fold of CV
    for (m in 1:n.models) {
      out<-nnet(type_ind~.,FGL1[-Ind[[k]],c(1:9,11)],linout = F, skip=F,size=as.integer(mod_par[2,m]),decay=mod_par[1,m],maxit=1000,trace=F)
      phat<-predict(out,FGL1[Ind[[k]],c(1:9,11)])
      yhat[Ind[[k]],m]<-apply(phat,1,function(x) which(x==max(x)))
    }
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum(y != x)/n)
} #end of j loop
MSE
MSEAve<- apply(MSE,2,mean); MSEAve #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); MSEsd   #SD of mean square CV error
##The best model in terms of the minimum MSEAve or the maximum r2.
min(MSEAve)
##Return the index of the minimum MSEAve or the maximum r2.
which(MSEAve==min(MSEAve))
##The optimal lambda and number of hidden nodes
mod_par[,which(MSEAve==min(MSEAve))]

##Do a CV in smaller interval of lambda and number of hidden nodes again.
Nrep<-10 #number of replicates of CV
K<-10  #K-fold CV on each replicate
n.lam = 19 #number of lambda
n.num_hidnode = 5 #number of different numbers of hidden nodes
n.models = n.lam*n.num_hidnode #number of different models to fit
n=nrow(FGL1)
y<-FGL1$type_ind
yhat=matrix(0,n,n.models)
lam_seq = c(seq(0.01,0.09,0.01),seq(0.1,1,0.1))
num_hidnode_seq = seq(12,20,2) 
mod_par=matrix(c(rep(lam_seq,times=1,each=n.num_hidnode),rep(num_hidnode_seq,times=n.lam,each=1)),2,n.models,byrow = T)#Store the model parameters: lambda and the number of nodes in hidden layer
MSE<-matrix(0,Nrep,n.models)
for (j in 1:Nrep) {
  print(c(0,0,0,j))#Print out the index of replicates of CV
  Ind<-CVInd(n,K)
  for (k in 1:K) {
    print(k)#Print out the index of different fold of CV
    for (m in 1:n.models) {
      out<-nnet(type~.,FGL1[-Ind[[k]],c(1:9,11)],linout = F, skip=F,size=as.integer(mod_par[2,m]),decay=mod_par[1,m],maxit=1000,trace=F)
      phat<-predict(out,FGL1[Ind[[k]],c(1:9,11)])
      yhat[Ind[[k]],m]<-apply(phat,1,function(x) which(x==max(x)))
    }
  } #end of k loop
  MSE[j,]=apply(yhat,2,function(x) sum(y != x)/n)
} #end of j loop
MSE
MSEAve<- apply(MSE,2,mean); MSEAve #averaged mean square CV error
MSEsd <- apply(MSE,2,sd); MSEsd   #SD of mean square CV error
r2<-1-MSEAve/var(y); r2  #CV r^2
##The best model in terms of the minimum MSEAve or the maximum r2.
min(MSEAve)
max(r2)
##Return the index of the minimum MSEAve or the maximum r2.
which(MSEAve==min(MSEAve))
which(r2==max(r2))
##The optimal lambda and number of hidden nodes
par_best<-mod_par[,which(MSEAve==min(MSEAve))]

##Fit the best nnet model
out<-nnet(type~.,FGL1[,c(1:10)],linout = F, skip=F,size=as.integer(par_best[2]),decay=par_best[1],maxit=1000,trace=F)##type is a factor
phat<-predict(out,FGL1)
yhat<-apply(phat,1,function(x) which(x==max(x)))
e.nnet<-sum(yhat!=y)/n

##(b)10-fold CV to find the best tree model for classifying the class type.
library(rpart)
control <- rpart.control(minbucket = 1, cp = 0.0001, maxsurrogate = 0, usesurrogate = 0, xval = 10)
par(mfrow=c(1,1),pin=c(4,4),mgp=c(2,1,0))
FGL1.tr <- rpart(type ~ .,FGL1[,c(1:10)], method = "class", control = control)
plotcp(FGL1.tr)  #plot of CV r^2 vs. size
printcp(FGL1.tr)  #same info is in df_std.tr$cptable
#prune back to optimal size, according to plot of CV 1-r^2
FGL1.tr1 <- prune(FGL1.tr, cp=0.0108696)  #approximately the best size pruned tree
FGL1.tr1$variable.importance#The importance of each predictors
FGL1.tr1$cptable[nrow(FGL1.tr1$cptable),] #shows training and CV 1-r^2, and other things
#prune and plot a little smaller tree than the optimal one, just for display
FGL1.tr2 <- prune(FGL1.tr, cp=0.0108696)  #bigger cp gives smaller size tree
FGL1.tr2
par(cex=.5); plot(FGL1.tr2, uniform=F); text(FGL1.tr2, use.n = T); par(cex=1)
##
yhat<-apply(predict(FGL1.tr1),1,function(x) which(x==max(x)))
e.tr<-sum(FGL1$type_ind!=yhat)/n

##(c)Fit multinomial results and discuss it
##Answer: 
FGL1.multinom<-multinom(type~.,FGL1[,c(1:10)])
yhat<-predict(FGL1.multinom,FGL1[,c(1:10)])
e.log<-sum(FGL1$type!=yhat)/n

##(d)Compare the three models from parts (a)-(c).
##Answer: The neural network has the best predictive ability but not very interpretable. Classification tree has very good interpretability, but the predictive ability is not as good as that of neural network. The multinomial regression has the worse predictive ability and the interpretability is better than neural network.

