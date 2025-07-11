# -------------------------------------------------------------------
# R code to cross‐plot "time vs pso_iter" and "fnval vs pso_iter"
# for each of three functions (rosenbrock, rastrigin, ackley).
# -------------------------------------------------------------------

# (1) Specify the path to your data file.  If it's in your working directory, you
#     can just give the filename.  Adjust as necessary.
data_file <- "5d_results.tsv"  
# e.g. data_file <- "pso_results_all_functions.txt"
# -------------------------------------------------------------------
# Step 2: Read the data, forcing no column to be used as row.names.
# We assume tabs ("\t") separate your fields and that there is a header row.
df <- read.delim(
  file      = data_file,
  header    = TRUE,
  sep       = "\t",
  stringsAsFactors = FALSE,
  row.names = NULL    # <- this prevents R from trying to guess row names
)

# -------------------------------------------------------------------
# Step 3: Double‑check that everything looks sane
str(df)
# Make sure 'function' is character/factor, 'pso_iter', 'time', 'fnval' are numeric, etc.

# If any of those columns came in as character, coerce them now:
df$fun <- as.factor(df$fun)
df$pso_iter <- as.numeric(df$pso_iter)
df$time     <- as.numeric(df$time)
df$fnval    <- as.numeric(df$fnval)

# -------------------------------------------------------------------
# Step 4: Split by function and recreate the plots exactly as before.
func_list <- split(df, df$fun)

plot_by_function <- function(subdf, out_prefix) {
  fname_time  <- paste0(out_prefix, "_time_vs_pso_iter.png")
  fname_fnval <- paste0(out_prefix, "_fnval_vs_pso_iter.png")

  # (a) time vs pso_iter
  png(filename = fname_time, width = 600, height = 450, res = 100)
  plot(
    subdf$pso_iter, subdf$time,
    type = "b", pch = 16,
    xlab = "Number of PSO Iterations",
    ylab = "Time",
    main = paste0(out_prefix, ": Time vs PSO Iterations")
  )
  grid()
  dev.off()

  # (b) fnval vs pso_iter
  png(filename = fname_fnval, width = 600, height = 450, res = 100)
  plot(
    subdf$pso_iter, subdf$fnval,
    type = "b", pch = 16,
    xlab = "Number of PSO Iterations",
    ylab = "Function Value (fnval)",
    main = paste0(out_prefix, ": fnval vs PSO Iterations")
  )
  grid()
  dev.off()
}

for (fn in names(func_list)) {
  plot_by_function(func_list[[fn]], out_prefix = fn)
}

# After running this, you’ll get six PNGs:
#   rosenbrock_time_vs_pso_iter.png
#   rosenbrock_fnval_vs_pso_iter.png
#   rastrigin_time_vs_pso_iter.png
#   rastrigin_fnval_vs_pso_iter.png
#   ackley_time_vs_pso_iter.png
#   ackley_fnval_vs_pso_iter.png
# -------------------------------------------------------------------

