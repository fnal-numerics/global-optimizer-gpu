# Read our TSV data and add an algorithm label.
common_names <- c("function", "opt", "time", "error", "minima", "coordinates")
our_data <- read.table("10d_results.tsv", header = TRUE, sep = "", stringsAsFactors = FALSE)
names(our_data)[1:6] <- common_names
our_data$algorithm <- "Ours"

# Read the ParallelSyncPSOKernel TSV data and add its label.
sync_data <- read.table("./related/psogpu/data/ParallelSyncPSOKernel/multistart_10dim_results.tsv",
                        header = FALSE, sep = "", fill = TRUE, stringsAsFactors = FALSE)
colnames(sync_data) <- common_names#c("function", "opt", "time", "error", "minima", "coordinates")
#sync_data <- read.delim("./related/psogpu/data/ParallelSyncPSOKernel/multistart_10dim_results.tsv", 
#                        header = TRUE, stringsAsFactors = FALSE)
sync_data$algorithm <- "ParallelSyncPSO"

# Read the ParallelPSOKernel TSV data and add its label.
kernel_data <- read.table("./related/psogpu/data/ParallelPSOKernel/multistart_10dim_results.tsv",
                          header = TRUE, sep="",stringsAsFactors = FALSE)
colnames(kernel_data) <- common_names
kernel_data$algorithm <- "ParallelPSOKernel"

# Read the Async data and add its label.
async_data <- read.table("./related/psogpu/data/ParallelPSOKernel/async_multistart_10dim_results.tsv",
                         header = TRUE, sep="",stringsAsFactors = FALSE)
colnames(async_data) <- common_names
async_data$algorithm <- "Async"

# Combine the datasets.
data_all <- rbind(our_data, sync_data, kernel_data, async_data)

# Convert relevant columns to numeric.
data_all$opt    <- as.numeric(data_all$opt)
data_all$time   <- as.numeric(data_all$time)
data_all$error  <- as.numeric(data_all$error)
data_all$minima <- as.numeric(data_all$minima)

# Aggregate numerical values by function, algorithm, and optimization number.
data_all_agg <- aggregate(cbind(time, error, minima) ~ `function` + algorithm + opt,
                          data = data_all, FUN = mean)

# Get unique functions.
fnames <- unique(data_all_agg$`function`)

# Define color mapping for algorithms.
cols <- c("Ours" = "blue", "ParallelSyncPSO" = "red", "ParallelPSOKernel" = "green", "Async" = "purple")

# Open a PDF device for plotting.
pdf("performance_plots.pdf", width = 10, height = 8)

# For each function, create two plots.
for (fn in fnames) {
  d <- data_all_agg[data_all_agg$`function` == fn, ]
  
  # Plot Optimization Time vs. Number of Optimizations.
  plot(NULL, xlim = range(d$opt), ylim = range(d$time), log = "x",
       xlab = "Number of Optimizations (log scale)", ylab = "Time (sec)",
       main = paste(fn, "- Time vs. Optimizations"))
  for (alg in sort(unique(d$algorithm))) {
    d_alg <- d[d$algorithm == alg, ]
    d_alg <- d_alg[order(d_alg$opt), ]
    lines(d_alg$opt, d_alg$time, type = "b", pch = 16, col = cols[alg])
  }
  legend("topright", legend = sort(unique(d$algorithm)),
         col = cols[sort(unique(d$algorithm))], lty = 1, pch = 16)
  
  # Plot Euclidean Error vs. Number of Optimizations.
  plot(NULL, xlim = range(d$opt), ylim = range(d$error), log = "x",
       xlab = "Number of Optimizations (log scale)", ylab = "Error",
       main = paste(fn, "- Error vs. Optimizations"))
  for (alg in sort(unique(d$algorithm))) {
    d_alg <- d[d$algorithm == alg, ]
    d_alg <- d_alg[order(d_alg$opt), ]
    lines(d_alg$opt, d_alg$error, type = "b", pch = 16, col = cols[alg])
  }
  legend("topright", legend = sort(unique(d$algorithm)),
         col = cols[sort(unique(d$algorithm))], lty = 1, pch = 16)
}

dev.off()

cat("Plots saved to 'performance_plots.pdf'\n")

