#!/usr/bin/env Rscript

fun <- "rosenbrock"

filename <- paste0(fun,"2d_particledata.tsv")
df <- read.delim(filename, header = TRUE, stringsAsFactors = FALSE)
df$status <- factor(df$status, levels = c(0, 1, 2),
                    labels = c("Surrendered", "Converged", "StoppedEarly"))

status_levels <- levels(df$status)
cols <- c("green", "red", "blue")
pch_vals <- c(4, 16, 17)
col_by_status <- cols[as.integer(df$status)]
pch_by_status <- pch_vals[as.integer(df$status)]

path <- paste0(fun, "/iter_distribution.pdf")
# iteration distribution with status densities
pdf(path, width = 7, height = 5)
all_iter <- df$iter
breaks <- seq(min(all_iter) - 0.5, max(all_iter) + 0.5, by = 1)
hist(all_iter, breaks = breaks, freq = FALSE, col = "lightgray", border = "white", xlab = "iter", ylab = "Density", main   = "")
for (i in seq_along(status_levels)) {
  group_iter <- df$iter[df$status == status_levels[i]]
  if (length(group_iter) > 1) {
    dens <- density(group_iter, from = min(breaks), to = max(breaks))
    lines(dens$x, dens$y, col = cols[i], lwd = 2)
  }
}
legend("topright", legend = status_levels, col = cols, lwd = 2, bty = "n")
dev.off()

# coordinates scatter
path <- paste(fun, "coord_scatter.pdf", sep = "/")
pdf(path, width = 6, height = 6)
plot(df$coord_0, df$coord_1, col = col_by_status, pch = pch_by_status,
     xlab = "coord_0", ylab = "coord_1")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")
dev.off()

# boxplots of fval and norm
path <- paste0(fun,"/boxplots_fval_norm.pdf")
pdf(path, width = 8, height = 4)
par(mfrow = c(1, 2))
boxplot(fval ~ status, data = df, col = cols, border = "black",
        main = "fval by Status", xlab = "Status", ylab = "fval")
boxplot(norm ~ status, data = df, col = cols, border = "black",
        main = "norm by Status", xlab = "Status", ylab = "norm")
par(mfrow = c(1, 1))
dev.off()

# scatter of iter vs fval
path <- paste0(fun, "/iter_vs_fval.pdf")
pdf(path, width = 7, height = 5)
plot(df$iter, df$fval, col = col_by_status, pch = pch_by_status,
     xlab = "iter", ylab = "fval")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")
dev.off()

# scatter of norm vs fval
path <- paste(fun,"norm_vs_fval.pdf", sep = "/")
pdf(path, width = 7, height = 5)
plot(df$norm, df$fval, col = col_by_status, pch = pch_by_status,
     xlab = "norm", ylab = "fval")
legend("topright", legend = status_levels, col = cols, pch = pch_vals, bty = "n")
dev.off()

cat("\nCount by Status:\n")
print(table(df$status))

cat("\nCorrelation Matrix (iter, fval, norm, coord_0, coord_1):\n")
num_cols <- df[, c("iter", "fval", "norm", "coord_0", "coord_1")]
print(round(cor(num_cols), 3))

