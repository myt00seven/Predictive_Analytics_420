# Homework 1
# Yintai Ma
# Feb 9 2017
#
require(gdata)
data <- read.xls("/Users/myt007/Google Drive/NU/2017 Winter/Predictive Analytics II/HW2/HW2_data.xls",sheet=1)
data
data$logcost <- log(data$cost)/log(10)

# 1a
#Prob 1)
##(a)Fit a linear model and discuss the predictive power.
##Answer:  I take log10 transform to the cost, the response variable, and then fit the model with all predictors unchanged. The $R^2$ is $0.5831$. The model with every predicots standardized has R-square $0.5527$. For those predicots that seems to be heavily tailed, I apply log transform on these predictos to see if that will help to imporve the model. Then the $R^2$ increases to $0.658$. Generally, those predictors significant before transform are also significant afterwards. 

##The histogram of each columns
df<-read.xls("./HW2_data.xls",sheet=1,header=TRUE)
par(mfrow=c(3,3))
for (i in seq(2,10)) hist(df[[i]],breaks=30,xlab=names(df)[i])
mod1<-lm(log10(cost)~.,data = df[-1])
summary(mod1)
##Also, I tried to standardize each variable to see effect.
df_std<-df
df_std$cost <- log10(df_std$cost)
df_std[2:10]<-sapply(df_std[2:10], function(x) (x-mean(x))/sd(x))
mod2<-lm(cost~.,data=df_std[-1])
summary(mod2)

## here is the histogram showing the relation between the log response and predictors.

par(mfrow=c(3,3))
for (i in seq(2,10)) {
  if (i %in% c(2,5,6,7,9)){
    hist(log10(df[[i]]+1),breaks=30,xlab=paste(names(df)[i],'log',sep='-'))  
  }
  if (i==3){
    hist(log10(max(df[[i]])+1-df[[i]]),breaks=30,xlab = paste(names(df)[i],'log','special',sep='-'))
  }
  if (i %in% c(4,8,10)){
    hist(df[[i]],breaks=30,xlab=names(df)[i])
  }
  
}
mod3<-lm(log10(cost)~1+log10(max(age)+1-age)+gend+log10(intvn+1)+log10(drugs+1)+log10(ervis+1)+comp+log10(comorb+1)+dur,data = df)#no matter use log10(age+1) or log10(I(age+1)), the result is the same.
summary(mod3)

## Prob 1(b)
### Answer: It seems that the interventions has the largest value of coefficient in each of three models I fitted above. Therfore, the interventions has the most influence on the cost.

## Prob 1(c)
### Answer: I use the second model as the observatino object to see if there is any problem. I have drawn residual plot versus y_hat and each variables to see if there is any indication for nonlinearity. For the residual plots vs. predictors, it seems that most of them has no clear signal showing nonlinearity. The intervention and age seems to have some degree of positive correlation with the residual. For the residual plot vs. y_hat, which is the fitted values. It seems that there is a linear relationship between the residual and the fitted value. For the last plot, the residual vs. log(cost), we can observe positive correlation between the residual and the log(cost) when the log(cost) is large. 

par(mfrow=c(1,4))
plot(df$intvn,resid(mod2),ylab="Residuals",xlab="intvn")
plot(df$comp,resid(mod2),ylab="Residuals",xlab="comp")
plot(df$drugs,resid(mod2),ylab="Residuals",xlab="drugs")
plot(df$ervis,resid(mod2),ylab="Residuals",xlab="ervis")

plot(df$age,resid(mod2),ylab="Residuals",xlab="age")
plot(df$gend,resid(mod2),ylab="Residuals",xlab="gend")
plot(df$comorb,resid(mod2),ylab="Residuals",xlab="comorb")
plot(df$dur,resid(mod2),ylab="Residuals",xlab="dur")

par(mfrow=c(1,1))
plot(mod2$fitted.values,resid(mod2),ylab="Residuals",xlab="y_hat")

par(mfrow=c(1,1))
plot(log10(df$cost),resid(mod2),ylab="Residuals",xlab="log(cost)",main="Ischemic heart disease-standardized predictors with log(cost)-lm")
abline(0, 0)

# Problem 2
## Prob 2(a) 
### Q:Use 10-fold cross-validation to find the best combination of shrinkage parameter λ and number of hidden nodes
### Answer: I conducted 10-fold CV to find the best parameters. The best choice is:


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
