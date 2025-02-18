library(ggplot2)
library(dplyr)
library(viridis)
library(grid)
# Get the file path from command line arguments
args <- commandArgs(trailingOnly = TRUE)

# Check if the argument is provided
if (length(args) < 1) {
  stop("Error: Please provide the file path as an argument.")
}

file_path <- args[1]

# Read and ensure numeric columns
step_data <- read.table(file_path, header = TRUE, stringsAsFactors = FALSE)
# Read and ensure numeric columns
#step_data <- read.table("optimization_steps.txt", header = TRUE, stringsAsFactors = FALSE)
step_data$OptIndex <- as.numeric(step_data$OptIndex)
step_data$Step <- as.numeric(step_data$Step)
step_data$X_0 <- as.numeric(step_data$X_0)
step_data$X_1 <- as.numeric(step_data$X_1)

# Filter out only exact (0,0) trailing steps
# First find the last nonzero step for each OptIndex
filtered_data <- step_data %>%
  group_by(OptIndex) %>%
  mutate(is_zero = (X_0 == 0 & X_1 == 0)) %>%
  mutate(last_nonzero_step = max(Step[!is_zero], na.rm=TRUE)) %>%
  filter(Step <= last_nonzero_step) %>%
  ungroup()

# Arrange data by OptIndex and Step
filtered_data <- filtered_data %>% arrange(OptIndex, Step)

# Identify start, intermediate, end points
start_points <- filtered_data %>%
  group_by(OptIndex) %>%
  filter(Step == min(Step)) %>%
  ungroup()

end_points <- filtered_data %>%
  group_by(OptIndex) %>%
  filter(Step == max(Step)) %>%
  ungroup()

intermediate_points <- filtered_data %>%
  group_by(OptIndex) %>%
  filter(Step != min(Step) & Step != max(Step)) %>%
  ungroup()

# Create segments from step to the next
segments_data <- filtered_data %>%
  group_by(OptIndex) %>%
  mutate(Xend = lead(X_0), Yend = lead(X_1)) %>%
  filter(!is.na(Xend)) %>%
  ungroup()

# Rastrigin grid
x_min <- -5.12; x_max <- 5.12; y_min <- -5.12; y_max <- 5.12
num_points <- 500
x_seq <- seq(x_min, x_max, length.out = num_points)
y_seq <- seq(y_min, y_max, length.out = num_points)
grid <- expand.grid(x = x_seq, y = y_seq)
rastrigin_function <- function(x, y, A = 10) {
  A * 2 + (x^2 - A * cos(2 * pi * x)) + (y^2 - A * cos(2 * pi * y))
}
grid$z <- rastrigin_function(grid$x, grid$y)

# Plot
p <- ggplot() +
  geom_raster(data = grid, aes(x = x, y = y, fill = z)) +
  scale_fill_viridis(option = "plasma", name = "Function Value") +

  geom_segment(data = segments_data,
               aes(x = X_0, y = X_1, xend = Xend, yend = Yend, group = OptIndex),
               arrow = arrow(length = unit(0.1, "cm")), color = "blue", alpha = 0.7) +

  geom_point(data = start_points, aes(x = X_0, y = X_1),
             color = "red", size = 1.5, alpha = 0.9) +
  geom_point(data = intermediate_points, aes(x = X_0, y = X_1),
             color = "deepskyblue", size = 1, alpha = 0.7) +
  geom_point(data = end_points, aes(x = X_0, y = X_1),
             color = "green", size = 1.5, alpha = 0.9) +

  labs(
    title = "Step-by-Step Optimization Trajectories on Rastrigin Function (No Zero Rows)",
    x = "X-axis",
    y = "Y-axis"
  ) +
  theme_minimal() +
  coord_fixed() +
  theme(legend.position = "right")

print(p)
ggsave("optimization_trajectories_cleaned.pdf", plot = p, width = 8, height = 6, dpi = 300, device = cairo_pdf)
cat("Plot saved to optimization_trajectories_cleaned.pdf\n")

