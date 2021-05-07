N = 200
J = 200

sim1 <- function(N, J, K){
  F.matr <- matrix(runif(N*K,-1,1), N, K);
  F.matr <- cbind(rep(1,N), F.matr)
  A.matr <- matrix(runif(J*(K+1),-1,1), J, K+1);                 
  
  temp = F.matr %*% t(A.matr);
  rate = exp(temp);
  resp = matrix(0, N, J);
  resp[] = rpois(N*J, rate)
  
  list(resp = resp)
}

set.seed(1)
data = sim1(N, J, 3)
eigen.res = eigen(cov(data$resp))

pdf(file = "scree.pdf", height = 5, width = 5)
plot(eigen.res$values[1:15], xlab = "eigenvalues", ylab = "eigenvalue")

dev.off()