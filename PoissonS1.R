jobid<-Sys.getenv("SLURM_ARRAY_TASK_ID");
jobid=as.numeric(jobid);

set.seed(jobid);

library(Rcpp)
library(RcppArmadillo)
Rcpp::sourceCpp("confirm_jmle_omp_poisson_with_intercept_missing.cpp")


J.vec = c(rep(100,2), rep(200,2), rep(300,2), rep(400,2))
N.vec = J.vec * rep(c(1,5), times = 4)


like.matr <- matrix(0, 24,5)
u.vec <- rep(0, 24)
Kest.vec <- rep(0, 24)
M.est <- rep(0, 24)
time.vec <- rep(0, 24)



sim1 <- function(N, J, K){
  F.matr <- matrix(runif(N*K,-1,1), N, K);
  F.matr <- cbind(rep(1,N), F.matr)
  A.matr <- matrix(runif(J*(K+1),-1,1), J, K+1);                 
  
  temp = F.matr %*% t(A.matr);
  resp = matrix(0, N, J);
  resp[] = rpois(N*J, lambda = exp(temp))
  
  list(resp = resp, F.matr = F.matr, A.matr = A.matr,  M = temp)
}

missing.fun <- function(type, N, J, F.matr){
  if(type == 1){
    Omega = matrix(0, N, J)
  }else if (type ==2){
    Omega = matrix(0, N, J);
    Omega[] = rbinom(N*J, 1, 0.5);
  }else{
    temp = F.matr[,1]%*% t(rep(1, J));
    prob = 1/ 1/(1+exp(-temp));
    Omega = matrix(0, N, J);
    Omega[] = rbinom(N*J, 1, prob);
  }
  
  Omega
} 

JL <- function(resp, theta, A, nonmis_ind){
  temp = theta%*% t(A)
  temp1 = resp * temp - exp(temp)  
  lik = sum(temp1[nonmis_ind])
  list(lik = lik, M = temp)
}


z = 0 
for(i in 1:8){
  N = N.vec[i]
  J = J.vec[i]
  data = sim1(N, J, 3)
  Mtrue = data$M
  theta0 = cbind(data$F.matr, matrix(runif(N*2, -0.1,0.1), N,2))
  A0 = cbind(data$A.matr, matrix(runif(J*2, -0.1,0.1), J, 2))
  
  for(j in 1:3){
    #print(j)
    time = proc.time()
    z = z+1;
    resp = data$resp;
    Omega = missing.fun(j, N, J, data$F.matr[,-1])
    resp[Omega==1] = NA
    
    nonmis_ind = !is.na(resp)
    resp[is.na(resp)] = 0
    
    n = sum(Omega==0)
    jml.res1 = confirm_CJMLE_poisson_cpp(resp,  nonmis_ind, theta0[,1:2], A0[,1:2], matrix(TRUE, J, 2), C = 3, tol = .1/n)
    jml.res2 = confirm_CJMLE_poisson_cpp(resp,  nonmis_ind, theta0[,1:3], A0[,1:3], matrix(TRUE, J, 3), C = 3, tol = .1/n)
    jml.res3 = confirm_CJMLE_poisson_cpp(resp,  nonmis_ind, theta0[,1:4], A0[,1:4], matrix(TRUE, J, 4), C = 3, tol = .1/n)
    jml.res4 = confirm_CJMLE_poisson_cpp(resp,  nonmis_ind, theta0[,1:5], A0[,1:5], matrix(TRUE, J, 5), C = 3, tol = .1/n)
    jml.res5 = confirm_CJMLE_poisson_cpp(resp,  nonmis_ind, theta0[,1:6], A0[,1:6], matrix(TRUE, J, 6), C = 3, tol = .1/n)
    
    
    jml1 = JL(resp, jml.res1$theta, jml.res1$A,nonmis_ind)
    jml2 = JL(resp, jml.res2$theta, jml.res2$A,nonmis_ind)
    jml3 = JL(resp, jml.res3$theta, jml.res3$A,nonmis_ind)
    jml4 = JL(resp, jml.res4$theta, jml.res4$A,nonmis_ind)
    jml5 = JL(resp, jml.res5$theta, jml.res5$A,nonmis_ind)
    
    like.matr[z,1] = jml1$lik
    like.matr[z,2] = jml2$lik
    like.matr[z,3] = jml3$lik
    like.matr[z,4] = jml4$lik
    like.matr[z,5] = jml5$lik
    u.vec[z] = N * log(n/N)
    JIC.vec <- -2*like.matr[z,] + (1:5) * u.vec[z]
    
    Kest.vec[z] = (1:5)[which.min(JIC.vec)] 
    
    M.err1 = sqrt(mean((jml3$M - Mtrue)^2))
    M.err2 = sqrt(mean((jml4$M - Mtrue)^2))
    M.err3 = sqrt(mean((jml5$M - Mtrue)^2))
    M.est[z] = max(c(M.err1, M.err2, M.err3))
    time.vec[z] = proc.time() - time
  }
}

filename  = paste("sim3res", jobid, ".Rdata", sep="")
save(like.matr, u.vec,Kest.vec, M.est, time.vec, file = filename)

