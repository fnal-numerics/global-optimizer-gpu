data_file <- "5d_results.tsv"

df <- read.delim(
  file           = data_file,
  header         = TRUE,
  sep            = "\t",
  stringsAsFactors = FALSE,
  row.names      = NULL,
  quote          = "",
  comment.char   = "",
  fill           = TRUE,
  check.names    = FALSE
)

# Drop unused 'N' and coordinate columns
df$N <- NULL
# df$N <- NULL
df$fun      <- as.factor(df$fun)
df$iter     <- as.numeric(df$iter)
df$pso_iter <- as.numeric(df$pso_iter)
df$time     <- as.numeric(df$time)
df$error    <- as.numeric(df$error)
df$fnval    <- as.numeric(df$fnval)
# df$coordinates remains character


# -------------------------------------------------------------------
# 3) Split by “fun”
# -------------------------------------------------------------------
func_list <- split(df, df$fun)

# -------------------------------------------------------------------
# 4) Define helper to save two PDFs per function
# -------------------------------------------------------------------
plot_by_function_pdf <- function(subdf, out_prefix) {
  pdf_time  <- paste0(out_prefix, "_time_vs_pso_iter.pdf")
  pdf_fnval <- paste0(out_prefix, "_fnval_vs_pso_iter.pdf")

  # (a) Time vs. PSO Iterations
  pdf(file = pdf_time, width = 5, height = 4.5) 
  plot(
    subdf$pso_iter, subdf$time,
    type = "b", pch = 16,
    xlab = "Number of PSO Iterations",
    ylab = "Time (ms)",
    #main = paste0(out_prefix, ": Time vs PSO Iters")
  )
  grid()
  dev.off()

  # (b) fnval vs. PSO Iterations
  pdf(file = pdf_fnval, width = 5, height = 4.5)
  plot(
    subdf$pso_iter, subdf$fnval,
    type = "b", pch = 16,
    xlab = "Number of PSO Iterations",
    ylab = "Function Value",
    #main = paste0(out_prefix, ": Function value vs PSO Iters")
  )
  grid()
  dev.off()
}

# -------------------------------------------------------------------
# 5) Loop and save PDFs
# -------------------------------------------------------------------
for (fn in names(func_list)) {
  plot_by_function_pdf(func_list[[fn]], out_prefix = fn)
}


