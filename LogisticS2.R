#Under the weak factor setting

jobid<-Sys.getenv("SLURM_ARRAY_TASK_ID");
jobid=as.numeric(jobid);

set.seed(jobid);

J.vec = c(rep(100,2), rep(200,2), rep(300,2), rep(400,2))
N.vec = J.vec * rep(c(1,5), times = 4)

like.matr <- matrix(0, 24,5)
u.vec <- rep(0, 24)
Kest.vec <- rep(0, 24)
M.est <- rep(0, 24)
time.vec <- rep(0, 24)

.libPaths("/home/yx_chen1988_gmail_com/R/x86_64-redhat-linux-gnu-library/3.6")
library(mirtjml)

sim1 <- function(N, J, K){
  F.matr <- matrix(runif(N*K,-2,2), N, K);
  F.matr[,K] <- runif(N, -0.8, 0.8)
  F.matr <- cbind(rep(1,N), F.matr)
  A.matr <- matrix(runif(J*(K+1),-2,2), J, K+1);                 
  
  temp = F.matr %*% t(A.matr);
  prob = 1/(1+exp(-temp));
  resp = matrix(0, N, J);
  resp[] = rbinom(N*J, 1, prob)
  
  list(resp = resp, F.matr = F.matr[,-1], A.matr = A.matr[,-1], d.vec = A.matr[,1], M = temp)
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


JL <- function(data, A, Theta, d){
  N = nrow(data)
  temp = Theta %*% t(A) + rep(1, N) %*% t(d)
  M = temp
  prob = 1/(1+exp(-temp))
  temp = data * log(prob) + (1-data) * log(1-prob);
  temp[is.na(temp)] = 0;
  list(lik = sum(temp), M = M)
  
}

z = 0 
for(i in 1:8){
  N = N.vec[i]
  J = J.vec[i]
  data = sim1(N, J, 3)
  Mtrue = data$M
  
  
  theta0 = cbind(data$F.matr, matrix(runif(N*2, -0.1,0.1), N,2))
  A0 = cbind(data$A.matr, matrix(runif(J*2, -0.1,0.1), J, 2))
  d0 = data$d.vec
  for(j in 1:3){
    time = proc.time()
    print(j)
    z = z+1;
    resp = data$resp;
    Omega = missing.fun(j, N, J, data$F.matr)
    resp[Omega==1] = NA
    n = sum(Omega==0)
    jml.res1 = mirtjml_expr(resp, 1,theta0 = matrix(theta0[,1], ncol = 1),  A0 =matrix(A0[,1], ncol = 1),d0 = d0,  tol = 0.1, cc = 5, print_proc = F)
    jml.res2 = mirtjml_expr(resp, 2, theta0 =theta0[,1:2], A0 = A0[,1:2],d0 = d0, tol = 0.1, cc = 5, print_proc = F)
    jml.res3 = mirtjml_expr(resp, 3,  theta0 =theta0[,1:3], A0 = A0[,1:3],d0 = d0,tol = 0.1, cc = 5, print_proc = F)
    jml.res4 = mirtjml_expr(resp, 4, theta0 =theta0[,1:4], A0 = A0[,1:4],d0 = d0, tol = 0.1, cc = 5, print_proc = F)
    jml.res5 = mirtjml_expr(resp, 5,  theta0 =theta0[,1:5], A0 = A0[,1:5],d0 = d0, tol = 0.1, cc = 5, print_proc = F)
  
    
    jml1 = JL(resp, jml.res1$A_hat, jml.res1$theta_hat, jml.res1$d_hat)
    jml2 = JL(resp, jml.res2$A_hat, jml.res2$theta_hat, jml.res2$d_hat)
    jml3 = JL(resp, jml.res3$A_hat, jml.res3$theta_hat, jml.res3$d_hat)
    jml4 = JL(resp, jml.res4$A_hat, jml.res4$theta_hat, jml.res4$d_hat)
    jml5 = JL(resp, jml.res5$A_hat, jml.res5$theta_hat, jml.res5$d_hat)
    
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


filename  = paste("sim2res", jobid, ".Rdata", sep="")
save(like.matr, u.vec,Kest.vec, M.est, time.vec, file = filename)




