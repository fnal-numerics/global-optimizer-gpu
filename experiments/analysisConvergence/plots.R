#!/usr/bin/env Rscript

# Read TSV
df <- read.delim("rosenbrock2d_particledata.tsv", header = TRUE, stringsAsFactors = FALSE)

# Convert status to factor with labels
df$status <- factor(df$status, levels = c(0, 1, 2),
                    labels = c("Surrendered", "Converged", "StoppedEarly"))

# Assign colors and plotting symbols by status
status_levels <- levels(df$status)
cols <- c("red", "darkgreen", "blue")
pch_vals <- c(4, 16, 17)
col_by_status <- cols[as.integer(df$status)]
pch_by_status <- pch_vals[as.integer(df$status)]

# Open PDF device
pdf("analysis_plots.pdf", width = 7, height = 7)

# 1) Histogram + density of iter
hist(df$iter, breaks = seq(min(df$iter) - 0.5, max(df$iter) + 0.5, by = 1),
     col = "skyblue", border = "black", main = "Distribution of Iterations",
     xlab = "iter", ylab = "Count")
dens <- density(df$iter)
lines(dens$x, dens$y * length(df$iter) * diff(hist(df$iter, plot = FALSE)$breaks[1:2]),
      col = "darkred", lwd = 1.5)

# 2) Scatter of coord_0 vs coord_1, colored by status
plot(df$coord_0, df$coord_1, col = col_by_status, pch = pch_by_status,
     xlab = "coord_0", ylab = "coord_1", main = "Final Coordinates by Status")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")

# 3) Boxplots of fval and norm by status side by side
par(mfrow = c(1, 2))
boxplot(fval ~ status, data = df, col = cols, border = "black",
        main = "fval by Status", xlab = "Status", ylab = "fval")
boxplot(norm ~ status, data = df, col = cols, border = "black",
        main = "norm by Status", xlab = "Status", ylab = "norm")
par(mfrow = c(1, 1))

# 4) Scatter of iter vs fval colored by status
plot(df$iter, df$fval, col = col_by_status, pch = pch_by_status,
     xlab = "iter", ylab = "fval", main = "iter vs. fval")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")

# 5) Scatter of norm vs fval colored by status
plot(df$norm, df$fval, col = col_by_status, pch = pch_by_status,
     xlab = "norm", ylab = "fval", main = "norm vs. fval")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")

# 6) Scatter of idx vs iter colored by status
plot(df$idx, df$iter, col = col_by_status, pch = pch_by_status,
     xlab = "idx", ylab = "iter", main = "idx vs. iter")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")

# Close PDF device
dev.off()

# Print summaries to console
cat("\nCount by Status:\n")
print(table(df$status))

cat("\nCorrelation Matrix (iter, fval, norm, coord_0, coord_1):\n")
num_cols <- df[, c("iter", "fval", "norm", "coord_0", "coord_1")]
print(round(cor(num_cols), 3))

