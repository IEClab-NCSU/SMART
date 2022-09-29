#Provide the path of the output.txt properly as file. This is the file converted to .txt from .csv format after running the comparison.sh
df <- data.frame(read.table(file='output.txt',na.string="NA",sep="\t",quote="",header=T))
df
summary(df)
# Strategy          Clustering    Encoding        KCs     
# Assessment-based:300   First-Level :200   TF    :300   Min.   : 10  
# Paragraph-based :300   Hybrid      :200   TF-IDF:300   1st Qu.: 50  
# Second-Level:200                Median :100  
# Mean   :102  
# 3rd Qu.:150  
# Max.   :200  
# AIC              BIC         No..of.unique.KCs
# Min.   :385638   Min.   :385692   Min.   : 5.00    
# 1st Qu.:393120   1st Qu.:393174   1st Qu.:10.00    
# Median :394805   Median :394859   Median :18.00    
# Mean   :394315   Mean   :394369   Mean   :21.36    
# 3rd Qu.:395936   3rd Qu.:395990   3rd Qu.:26.00    
# Max.   :397063   Max.   :397117   Max.   :92.00    
# No..of.unique.compound.KCs No.of.unique.compound.KCs.with.partial.KC.names
# Min.   : 0.000             Min.   :0.0000                                 
# 1st Qu.: 2.000             1st Qu.:0.0000                                 
# Median : 2.000             Median :1.0000                                 
# Mean   : 3.245             Mean   :0.9117                                 
# 3rd Qu.: 4.000             3rd Qu.:1.0000                                 
# Max.   :22.000             Max.   :9.0000   

model <- lm(AIC ~ Strategy + Clustering + Encoding + KCs, data=df)
summary(model)

# 
#   
# Call:
#   lm(formula = AIC ~ Strategy + Clustering + Encoding + KCs, data = df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -6791.1  -909.5   198.3  1089.4  3430.7 
# 
# Coefficients:
#   Estimate Std. Error  t value Pr(>|t|)    
# (Intercept)             394182.624    183.297 2150.515   <2e-16 ***
#   StrategyParagraph-based   -291.466    136.113   -2.141   0.0327 *  
#   ClusteringHybrid          1836.069    166.704   11.014   <2e-16 ***
#   ClusteringSecond-Level    2411.876    166.704   14.468   <2e-16 ***
#   EncodingTF-IDF            -303.667    136.113   -2.231   0.0261 *  
#   KCs                         -9.663      1.002   -9.647   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1667 on 594 degrees of freedom
# Multiple R-squared:  0.3578,	Adjusted R-squared:  0.3524 
# F-statistic:  66.2 on 5 and 594 DF,  p-value: < 2.2e-16

model1 <- lm(AIC ~ Clustering + Encoding + KCs, data=df)
summary(model1)

# Call:
#   lm(formula = AIC ~ Clustering + Encoding + KCs, data = df)
# 
# Residuals:
#   Min      1Q  Median      3Q     Max 
# -6645.3  -890.5   218.5  1090.2  3285.0 
# 
# Coefficients:
#   Estimate Std. Error  t value Pr(>|t|)    
# (Intercept)            394036.891    170.706 2308.275   <2e-16 ***
#   ClusteringHybrid         1836.069    167.205   10.981   <2e-16 ***
#   ClusteringSecond-Level   2411.876    167.205   14.425   <2e-16 ***
#   EncodingTF-IDF           -303.667    136.522   -2.224   0.0265 *  
#   KCs                        -9.663      1.005   -9.618   <2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 1672 on 595 degrees of freedom
# Multiple R-squared:  0.3529,	Adjusted R-squared:  0.3485 
# F-statistic: 81.11 on 4 and 595 DF,  p-value: < 2.2e-16

model2 <- lm(AIC ~ Clustering + KCs, data=df)
summary(model2)$coefficients

# Estimate Std. Error     t value     Pr(>|t|)
# (Intercept)            393885.057391 156.981425 2509.118888 0.000000e+00
# ClusteringHybrid         1836.069500 167.757912   10.944757 1.572044e-25
# ClusteringSecond-Level   2411.876500 167.757912   14.377125 1.874583e-40
# KCs                        -9.663175   1.008032   -9.586176 2.446188e-20
anova(model2)


# Analysis of Variance Table
# 
# Response: AIC
# Df     Sum Sq   Mean Sq F value    Pr(>F)    
# Clustering   2  634656877 317328439 112.757 < 2.2e-16 ***
#   KCs          1  258616825 258616825  91.895 < 2.2e-16 ***
#   Residuals  596 1677305929   2814272                      
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1