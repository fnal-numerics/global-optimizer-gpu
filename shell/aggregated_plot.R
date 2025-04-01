# Read our TSV data and add an algorithm label.
our_data <- read.table("main_10d_results.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)
our_data$algorithm <- "Our"

# Read the ParallelSyncPSOKernel TSV data and add its label.
sync_data <- read.table("../related/psogpu/data/ParallelSyncPSOKernel/multistart_highdim_results.tsv", header = TRUE, sep = "\t", stringsAsFactors = FALSE)
sync_data$algorithm <- "ParallelSyncPSO"

### also add ./data/ParallelPSOKernel/multistart_highdim_results.tsv as ParallelPSOKernel
# Combine the datasets.
data_all <- rbind(our_data, sync_data)

# Convert relevant columns to numeric.
data_all$opt    <- as.numeric(data_all$opt)
data_all$time   <- as.numeric(data_all$time)
data_all$error  <- as.numeric(data_all$error)
data_all$minima <- as.numeric(data_all$minima)

# Use backticks to refer to the "function" column.
fnames <- unique(data_all$`function`)

# Open a PDF device for plotting.
pdf("performance_plots.pdf", width = 10, height = 8)

# For each function, create two plots.
for (fn in fnames) {
  d <- data_all[data_all$`function` == fn, ]
  
  # Plot Optimization Time vs. Number of Optimizations.
  plot(NULL, xlim = range(d$opt), ylim = range(d$time), log = "x",
       xlab = "Number of Optimizations (log scale)", ylab = "Time (sec)",
       main = paste(fn, "- Time vs. Optimizations"))
  for (alg in unique(d$algorithm)) {
    d_alg <- d[d$algorithm == alg, ]
    d_alg <- d_alg[order(d_alg$opt), ]
    lines(d_alg$opt, d_alg$time, type = "b", pch = 16,
          col = ifelse(alg == "Our", "blue", "red"))
  }
  legend("topright", legend = unique(d$algorithm), col = c("blue", "red"), lty = 1, pch = 16)
  
  # Plot Euclidean Error vs. Number of Optimizations.
  plot(NULL, xlim = range(d$opt), ylim = range(d$error), log = "x",
       xlab = "Number of Optimizations (log scale)", ylab = "Error",
       main = paste(fn, "- Error vs. Optimizations"))
  for (alg in unique(d$algorithm)) {
    d_alg <- d[d$algorithm == alg, ]
    d_alg <- d_alg[order(d_alg$opt), ]
    lines(d_alg$opt, d_alg$error, type = "b", pch = 16,
          col = ifelse(alg == "Our", "blue", "red"))
  }
  legend("topright", legend = unique(d$algorithm), col = c("blue", "red"), lty = 1, pch = 16)
}

dev.off()

cat("Plots saved to 'performance_plots.pdf'\n")

