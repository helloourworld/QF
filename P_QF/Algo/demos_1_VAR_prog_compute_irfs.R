## Prep workspace
## https://alexchinco.com/impulse-response-functions-for-vars/
options(width=120, digits=6, digits.secs=6)
rm(list=ls())

library(foreign)
library(grid)
library(plyr)
library(ggplot2)
library(tikzDevice)
print(options('tikzLatexPackages'))
options(tikzLatexPackages =
        c("\\usepackage{tikz}\n",
          "\\usepackage[active,tightpage,psfixbb]{preview}\n",
          "\\PreviewEnvironment{pgfpicture}\n",
          "\\setlength\\PreviewBorder{0pt}\n",
          "\\usepackage{amsmath}\n",
          "\\usepackage{xfrac}\n"
          )
        )
setTikzDefaults(overwrite = FALSE)
print(options('tikzLatexPackages'))
library(reshape)
library(vars)
library(scales)   
library(zoo)   
library(matpow)   




## Define directories
scl.str.DAT_DIR  <- "./data/"
scl.str.FIG_DIR  <- "./figures/"


## Load CPI data
mat.df.CPI        <- read.csv(file             = paste(scl.str.DAT_DIR, "data--fred-quarterly-cpi.csv", sep = ""),
                              stringsAsFactors = FALSE
                              )
mat.df.CPI$DATE   <- as.Date(mat.df.CPI$DATE, format = "%Y-%m-%d")
names(mat.df.CPI) <- c("t", "cpi")
scl.int.CPI_LEN   <- dim(mat.df.CPI)[1]
mat.df.CPI$cpi    <- as.numeric(mat.df.CPI$cpi)
mat.df.CPI$cpiLag <- c(NA, mat.df.CPI$cpi[1:(scl.int.CPI_LEN-1)])
mat.df.CPI        <- mat.df.CPI[is.na(mat.df.CPI$cpiLag) == FALSE, ]






## Load HPI data
mat.df.HPI        <- read.csv(file             = paste(scl.str.DAT_DIR, "data--fred-quarterly-hpi.csv", sep = ""),
                              stringsAsFactors = FALSE
                              )
mat.df.HPI$DATE   <- as.Date(mat.df.HPI$DATE, format = "%Y-%m-%d")
names(mat.df.HPI) <- c("t", "hpi")
scl.int.HPI_LEN   <- dim(mat.df.HPI)[1]
mat.df.HPI$hpi    <- as.numeric(mat.df.HPI$hpi)
mat.df.HPI$hpiLag <- c(NA, mat.df.HPI$hpi[1:(scl.int.HPI_LEN-1)])
mat.df.HPI        <- mat.df.HPI[is.na(mat.df.HPI$hpiLag) == FALSE, ]







## Merge data
mat.df.DATA <- merge(mat.df.CPI, mat.df.HPI, by = c("t"))




## Demean variables
scl.flt.AVG_CPI    <- mean(mat.df.DATA$cpi)
mat.df.DATA$cpi    <- mat.df.DATA$cpi - scl.flt.AVG_CPI
mat.df.DATA$cpiLag <- mat.df.DATA$cpiLag - scl.flt.AVG_CPI

scl.flt.AVG_HPI    <- mean(mat.df.DATA$hpi)
mat.df.DATA$hpi    <- mat.df.DATA$hpi - scl.flt.AVG_HPI
mat.df.DATA$hpiLag <- mat.df.DATA$hpiLag - scl.flt.AVG_HPI






## Estimate CPI auto-regression
obj.lm.CPI_AR <- lm(cpi ~ 0 + cpiLag, data = mat.df.DATA)
summary(obj.lm.CPI_AR)

scl.flt.STD_EP <- sd(obj.lm.CPI_AR$residuals)






## Plot impulse response function for AR(1)
if (TRUE == TRUE) {
    mat.df.PLOT <- data.frame(h  = seq(0, 12),
                              dx = NA
                              )
    for (h in 0:12) {
        mat.df.PLOT$dx[h + 1] <- obj.lm.CPI_AR$coef^h * sd(obj.lm.CPI_AR$residuals)
    }
    
    theme_set(theme_bw())
    
    scl.str.RAW_FILE <- "plot--cpi-ar1-irf"
    scl.str.TEX_FILE <- paste(scl.str.RAW_FILE,'.tex',sep='')
    scl.str.PDF_FILE <- paste(scl.str.RAW_FILE,'.pdf',sep='')
    scl.str.AUX_FILE <- paste(scl.str.RAW_FILE,'.aux',sep='')
    scl.str.LOG_FILE <- paste(scl.str.RAW_FILE,'.log',sep='')
    
    tikz(file = scl.str.TEX_FILE, height = 2, width = 7, standAlone=TRUE)    
    obj.gg2.PLOT <- ggplot(data = mat.df.PLOT)      
    obj.gg2.PLOT <- obj.gg2.PLOT + scale_colour_brewer(palette="Set1")
    obj.gg2.PLOT <- obj.gg2.PLOT + geom_hline(yintercept = 0,
                                              linetype   = 4,
                                              size       = 0.50
                                              )
    obj.gg2.PLOT <- obj.gg2.PLOT + geom_path(aes(x = h,
                                                 y = dx
                                                 ),
                                             size = 1.25
                                             )
    obj.gg2.PLOT <- obj.gg2.PLOT + xlab("$h$ $\\scriptstyle [\\text{qtr}]$")
    obj.gg2.PLOT <- obj.gg2.PLOT + ylab("$\\mathrm{Imp}(h)$ $\\scriptstyle [\\sfrac{\\%}{\\text{qtr}}]$")
    obj.gg2.PLOT <- obj.gg2.PLOT + scale_x_continuous(limits = c(0, 12), breaks = c(0, 1, 2, 4, 8, 12))
    obj.gg2.PLOT <- obj.gg2.PLOT + theme(plot.margin       = unit(c(1,0.15,0.15,0.15), "lines"),
                                         axis.text         = element_text(size = 8),
                                         axis.title        = element_text(size = 10),
                                         plot.title        = element_text(vjust = 1.75),
                                         panel.grid.minor  = element_blank()
                                         )
    obj.gg2.PLOT <- obj.gg2.PLOT + ggtitle("Impulse-Response Function for $\\mathrm{AR}(1)$")    
    print(obj.gg2.PLOT)
    dev.off()
    

    system(paste('lualatex', file.path(scl.str.TEX_FILE)), ignore.stdout = TRUE)
    system(paste('rm ', scl.str.TEX_FILE, sep = ''))
    system(paste('mv ', scl.str.PDF_FILE, ' ', scl.str.FIG_DIR, sep = ''))
    system(paste('rm ', scl.str.AUX_FILE, sep = ''))
    system(paste('rm ', scl.str.LOG_FILE, sep = ''))

}






## Estimate CPI + HPI vector auto-regression
obj.lm.CPI_VAR <- lm(cpi ~ 0 + cpiLag + hpiLag, data = mat.df.DATA)
summary(obj.lm.CPI_VAR)


obj.lm.HPI_VAR <- lm(hpi ~ 0 + cpiLag + hpiLag, data = mat.df.DATA)
summary(obj.lm.HPI_VAR)




## Plot impulse response function for VAR
if (TRUE == TRUE) {
    mat.flt.GAMMA <- matrix(rbind(obj.lm.CPI_VAR$coef, obj.lm.HPI_VAR$coef), nrow = 2)
    mat.flt.SIGMA <- var(cbind(obj.lm.CPI_VAR$residuals, obj.lm.HPI_VAR$residuals))
    mat.flt.C_TRN <- chol(mat.flt.SIGMA)
    mat.flt.C     <- t(chol(mat.flt.SIGMA))

    ## Correct
    mat.df.PLOT1 <- data.frame(h  = seq(0, 12),
                               dx = NA,
                               dy = NA
                               )
    mat.df.PLOT1$dx[1] <- mat.flt.C[1,1]
    mat.df.PLOT1$dy[1] <- mat.flt.C[2,1]
    for (h in 1:12) {
        mat.flt.GAMMA_POW                <- matpow(mat.flt.GAMMA, h)$prod1
        mat.df.PLOT1[h+1, c("dx", "dy")] <- mat.flt.GAMMA_POW %*% mat.flt.C %*% matrix(c(1,0), ncol = 1)
    }
    names(mat.df.PLOT1) <- c("h", "$\\Delta x_{t+h}$", "$\\Delta y_{t+h}$")
    mat.df.PLOT1        <- melt(mat.df.PLOT1, c("h"))
    names(mat.df.PLOT1) <- c("h", "variable", "vCorrect")
    
    
    ## Naive
    mat.df.PLOT2 <- data.frame(h  = seq(0, 12),
                               dx = NA,
                               dy = NA
                               )
    mat.df.PLOT2$dx[1] <- sd(obj.lm.CPI_VAR$residuals)
    mat.df.PLOT2$dy[1] <- 0
    for (h in 1:12) {
        mat.flt.GAMMA_POW                <- matpow(mat.flt.GAMMA, h)$prod1
        mat.df.PLOT2[h+1, c("dx", "dy")] <- mat.flt.GAMMA_POW %*% matrix(c(sd(obj.lm.CPI_VAR$residuals),0), ncol = 1)
    }
    names(mat.df.PLOT2) <- c("h", "$\\Delta x_{t+h}$", "$\\Delta y_{t+h}$")
    mat.df.PLOT2        <- melt(mat.df.PLOT2, c("h"))
    names(mat.df.PLOT2) <- c("h", "variable", "vNaive")

    ## Merge
    mat.df.PLOT <- merge(mat.df.PLOT1, mat.df.PLOT2, by = c("h", "variable"))
    mat.df.PLOT <- mat.df.PLOT[order(mat.df.PLOT$variable, mat.df.PLOT$h),]
    
    
    theme_set(theme_bw())
    
    scl.str.RAW_FILE <- "plot--cpi-vs-hpi-var-irf"
    scl.str.TEX_FILE <- paste(scl.str.RAW_FILE,'.tex',sep='')
    scl.str.PDF_FILE <- paste(scl.str.RAW_FILE,'.pdf',sep='')
    scl.str.AUX_FILE <- paste(scl.str.RAW_FILE,'.aux',sep='')
    scl.str.LOG_FILE <- paste(scl.str.RAW_FILE,'.log',sep='')
    
    tikz(file = scl.str.TEX_FILE, height = 2, width = 7, standAlone=TRUE)    
    obj.gg2.PLOT <- ggplot(data = mat.df.PLOT)                       
    obj.gg2.PLOT <- obj.gg2.PLOT + scale_colour_brewer(palette="Set1")
    obj.gg2.PLOT <- obj.gg2.PLOT + geom_hline(yintercept = 0,
                                              linetype   = 4,
                                              size       = 0.50
                                              )
    ## obj.gg2.PLOT <- obj.gg2.PLOT + geom_path(aes(x     = h,
    ##                                              y     = vNaive,
    ##                                              group = variable
    ##                                              ),
    ##                                          size = 1.25,
    ##                                          colour = "red"
    ##                                          )
    obj.gg2.PLOT <- obj.gg2.PLOT + geom_path(aes(x     = h,
                                                 y     = vCorrect,
                                                 group = variable
                                                 ),
                                             size = 1.25
                                             )
    obj.gg2.PLOT <- obj.gg2.PLOT + xlab("$h$ $\\scriptstyle [\\text{qtr}]$")
    obj.gg2.PLOT <- obj.gg2.PLOT + ylab("$\\mathrm{Imp}_x(h)$ $\\scriptstyle [\\sfrac{\\%}{\\text{qtr}}]$")
    obj.gg2.PLOT <- obj.gg2.PLOT + scale_x_continuous(limits = c(0, 12), breaks = c(0, 1, 2, 4, 8, 12))
    obj.gg2.PLOT <- obj.gg2.PLOT + facet_wrap(~ variable, nrow = 1)
    obj.gg2.PLOT <- obj.gg2.PLOT + theme(plot.margin       = unit(c(1,0.15,0.15,0.15), "lines"),
                                         axis.text         = element_text(size = 8),
                                         axis.title        = element_text(size = 10),
                                         plot.title        = element_text(vjust = 1.75),
                                         panel.grid.minor  = element_blank()
                                         )
    obj.gg2.PLOT <- obj.gg2.PLOT + ggtitle("Impulse-Response Function for $\\mathrm{VAR}$")    
    print(obj.gg2.PLOT)
    dev.off()
    

    system(paste('lualatex', file.path(scl.str.TEX_FILE)), ignore.stdout = TRUE)
    system(paste('rm ', scl.str.TEX_FILE, sep = ''))
    system(paste('mv ', scl.str.PDF_FILE, ' ', scl.str.FIG_DIR, sep = ''))
    system(paste('rm ', scl.str.AUX_FILE, sep = ''))
    system(paste('rm ', scl.str.LOG_FILE, sep = ''))
}