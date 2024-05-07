
"""
@author: lyu
Backus, D., Foresi, S., Telmer, C., 1998. Discrete-time models of bond pricing. Tech. rep., National bureau
of economic research.
Vasicek, O., 1977. An equilibrium characterization of the term structure. Journal of financial economics 5,
177–188
"""

# replicate Figure 1 of Backus et al. (1998)
us_gvt <- data.frame(Maturity=c(1,3,6,9,12,24,36,48,60,84,120),
Mean=c(5.314,5.640,5.884,6.003,6.079,6.272,6.386,6.467,6.531,6.624,6.683),
St.Dev=c(3.064,3.143,3.178,3.182,3.168,3.124,3.087,3.069,3.056,3.043,3.013),
Skewness=c(0.886,0.858,0.809,0.776,0.730,0.660,0.621,0.612,0.599,0.570,0.532),
Kurtosis=c(0.789,0.691,0.574,0.480,0.315,0.086,-0.066,-0.125,-0.200,-0.349,-0.477),
Auto=c(0.976,0.981,0.982,0.982,0.983,0.986,0.988,0.989,0.990,0.991,0.992))
theta <- us_gvt$Mean[1]/1200
varphi <- 0.976
lambda <- -0.0824
delta <- lambda^2/2
sigma <- sqrt((us_gvt$St.Dev[1]/1200)^2*(1-varphi^2))

# here the paper " Discrete-time models of bond pricing"
# wrongly report sigma=0.005560 instead of 0.0005560
B <- c()
B[1] <- 1
for (i in 2:120){
    B[i] <- 1+B[i-1]*varphi
}
A <- c()
A[1] <- 0
for (i in 2:120){
    A[i] <- A[i-1]+delta+B[i-1]*(1-varphi)*theta-((lambda+B[i-1]*sigma)^2)/2
}
A <- A[c(1,3,6,9,12,24,36,48,60,84,120)]
B <- B[c(1,3,6,9,12,24,36,48,60,84,120)]

bond.ylds_exp <- (A+B*theta)*us_gvt$Maturity^-1
bond.ylds_exp <- bond.ylds_exp * 1200

{plot(us_gvt$Maturity, us_gvt$Mean,pch = 8, col = "blue", xlab = "Maturity in months", ylab = "Mean Yield(Annual Percentage)")
 lines(us_gvt$Maturity,bond.ylds_exp, type="l", col = "red")
legend("bottomright", legend = c("US T-sec yields", "Vasicek model mean yields"), pch = c(8, NA), col = c("blue", "red"), lty = c(0, 1))}

## 5.2
# simulate the yield curve over the next four periods
innovations <- c(0.55, -0.28, 1.78, 0.19)
sigma <- sqrt((us_gvt$St.Dev/1200)^2*(1-varphi^2))
z.t1 <- varphi*bond.ylds_exp[1]/1200 + (1-varphi)*theta+sigma[1]*innovations[1]
z.t2 <- varphi*z.t1 + (1-varphi)*theta+sigma[1]*innovations[2]
z.t3 <- varphi*z.t2 + (1-varphi)*theta+sigma[1]*innovations[3]
z.t4 <- varphi*z.t3 + (1-varphi)*theta+sigma[1]*innovations[4]
B <- c()
B[1] <- 1
for (i in 2:120){
    B[i] <- 1+B[i-1]*varphi
}
A <- c()
A[1] <- 0
for (i in 2:120){
    A[i] <- A[i-1]+delta+B[i-1]*(1-varphi)*theta-((lambda+B[i-1]*sigma)^2)/2
}
A <- A[c(1,3,6,9,12,24,36,48,60,84,120)]
B <- B[c(1,3,6,9,12,24,36,48,60,84,120)]
curve1 <- A + B*z.t1
curve2 <- A + B*z.t2
curve3 <- A + B*z.t3
curve4 <- A + B*z.t4
curve1 <- curve1*us_gvt$Maturity^-1
curve2 <- curve2*us_gvt$Maturity^-1
curve3 <- curve3*us_gvt$Maturity^-1
curve4 <- curve4*us_gvt$Maturity^-1
curve1 <- curve1*1200
curve2 <- curve2*1200
curve3 <- curve3*1200
curve4 <- curve4*1200
{plot(us_gvt$Maturity, curve1, type = "l", ylim = c(5.2, 7.5), col = "blue", lwd = 2, xlab = "Maturity", ylab="Yields")
lines(us_gvt$Maturity, curve2, col = "red", lwd = 2)
lines(us_gvt$Maturity, curve3, col = "green", lwd = 2)
lines(us_gvt$Maturity, curve4, col = "purple", lwd = 2)
legend("bottomright", legend = c("t+1", "t+2", "t+3", "t+4"), col = c("blue", "red", "green", "purple"))}
