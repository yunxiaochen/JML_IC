J.vec = c(10, 20, 30, 40, 50)
N.vec = 2 * J.vec


library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("confirm_jmle_omp_linear_with_intercept.cpp")
Kest.matr <- matrix(0, 100, 5)
KestAIC.matr <-matrix(0, 100, 5)
KestBIC.matr <-matrix(0, 100, 5)

sim1 <- function(N, J, K){
  F.matr <- matrix(runif(N*K,-2,2), N, K);
  F.matr <- cbind(rep(1,N), F.matr)
  A.matr <- matrix(runif(J*(K+1),-2,2), J, K+1);                 
  
  temp = F.matr %*% t(A.matr);
  resp = matrix(0, N, J);
  resp[] = temp + rnorm(N*J)
  
  list(resp = resp, F.matr = F.matr, A.matr = A.matr,  M = temp)
}

Baifun <- function(resp, k.vec){
  S <- cov(resp)
  l.vec <- eigen(S)$values
  n = nrow(resp)
  p = ncol(resp)
  
  temp = l.vec[p:1] 
  temp = cumsum(l.vec[p:1])/(1:p)
  lbar.vec = temp[(p-1):1]
  
  AIC.vec = k.vec
  BIC.vec = k.vec
  K = length(k.vec)
  for(i in 1:K){
    k = k.vec[i]
    AIC.vec[i] = n * sum(log(l.vec[1:k])) + n *(p-k)*log(lbar.vec[k]) + 2 * (k +1)*(p +1 - k/2)
    BIC.vec[i] = n * sum(log(l.vec[1:k])) + n *(p-k)*log(lbar.vec[k]) + log(n) * (k +1)*(p +1 - k/2)
    
  }
  
  list(AIC.vec = AIC.vec, BIC.vec=BIC.vec)
  
}
JL <- function(resp, theta, A){
  temp = theta%*% t(A)
  
  lik =sum(log(dnorm(resp, mean = temp, sd = 1)))
  
  list(lik = lik, M = temp)
}


like.vec = rep(0,5)
for(z in 1:100){
for(i in 1:5){
  N = N.vec[i]
  J = J.vec[i] 
  
  data = sim1(N, J, 3)
  
  res = Baifun(data$resp, 1:5)
  KestAIC.matr[z,i] = which.min(res$AIC.vec)
  KestBIC.matr[z,i] = which.min(res$BIC.vec)
  
  theta0 = cbind(data$F.matr, matrix(runif(N*2, -0.1,0.1), N,2))
  A0 = cbind(data$A.matr, matrix(runif(J*2, -0.1,0.1), J, 2))
  
  resp = data$resp
  
  jml.res1 = CJMLE_linear(resp,   theta0[,1:2], A0[,1:2], matrix(TRUE, J, 2), 1,C = 5, tol = 0.01/N/J, F)
  jml.res2 = CJMLE_linear(resp,   theta0[,1:3], A0[,1:3], matrix(TRUE, J, 3), 1,C = 5, tol = 0.01/N/J, F)
  jml.res3 = CJMLE_linear(resp,   theta0[,1:4], A0[,1:4], matrix(TRUE, J, 4), 1,C = 5, tol = 0.01/N/J, F)
  jml.res4 = CJMLE_linear(resp,   theta0[,1:5], A0[,1:5], matrix(TRUE, J, 5), 1,C = 5, tol = 0.01/N/J, F)
  jml.res5 = CJMLE_linear(resp,   theta0[,1:6], A0[,1:6], matrix(TRUE, J, 6), 1,C = 5, tol = 0.01/N/J, F)
  
  jml1 = JL(resp, jml.res1$theta, jml.res1$A)
  jml2 = JL(resp, jml.res2$theta, jml.res2$A)
  jml3 = JL(resp, jml.res3$theta, jml.res3$A)
  jml4 = JL(resp, jml.res4$theta, jml.res4$A)
  jml5 = JL(resp, jml.res5$theta, jml.res5$A)
  
  like.vec[1] = jml1$lik
  like.vec[2] = jml2$lik
  like.vec[3] = jml3$lik
  like.vec[4] = jml4$lik
  like.vec[5]  = jml5$lik
  u  = N * log(J)
  JIC.vec <- -2*  like.vec + (1:5) * u 
  
  Kest.matr[z,i] = (1:5)[which.min(JIC.vec)] 
}
}

colSums(Kest.matr==3)
colSums(Kest.matr>3)
colSums(Kest.matr<3)

colSums(KestAIC.matr==3)
colSums(KestAIC.matr>3)
colSums(KestAIC.matr<3)

colSums(KestBIC.matr==3)
colSums(KestBIC.matr>3)
colSums(KestBIC.matr<3)