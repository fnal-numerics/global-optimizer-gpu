# Ackley Function Version
library(ggplot2)
library(dplyr)
library(viridis)
library(grid)

# Get file path from command line args
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Error: Please provide the file path as an argument.")
file_path <- args[1]

# Read and convert columns to numeric
step_data <- read.table(file_path, header = TRUE, stringsAsFactors = FALSE)
step_data$OptIndex <- as.numeric(step_data$OptIndex)
step_data$Step    <- as.numeric(step_data$Step)
step_data$X_0     <- as.numeric(step_data$X_0)
step_data$X_1     <- as.numeric(step_data$X_1)

# Filter out trailing (0,0) steps
filtered_data <- step_data %>%
  group_by(OptIndex) %>%
  mutate(is_zero = (X_0 == 0 & X_1 == 0)) %>%
  mutate(last_nonzero_step = max(Step[!is_zero], na.rm = TRUE)) %>%
  filter(Step <= last_nonzero_step) %>%
  ungroup() %>%
  arrange(OptIndex, Step)

# Identify start, intermediate, end points
start_points <- filtered_data %>% group_by(OptIndex) %>% filter(Step == min(Step)) %>% ungroup()
end_points   <- filtered_data %>% group_by(OptIndex) %>% filter(Step == max(Step)) %>% ungroup()
intermediate_points <- filtered_data %>% 
  group_by(OptIndex) %>% 
  filter(Step != min(Step) & Step != max(Step)) %>% 
  ungroup()

# Create segments for trajectory lines
segments_data <- filtered_data %>%
  group_by(OptIndex) %>%
  mutate(Xend = lead(X_0), Yend = lead(X_1)) %>%
  filter(!is.na(Xend)) %>%
  ungroup()

# Ackley grid parameters and function
x_min <- -5; x_max <- 5; y_min <- -5; y_max <- 5; num_points <- 500
x_seq <- seq(x_min, x_max, length.out = num_points)
y_seq <- seq(y_min, y_max, length.out = num_points)
grid <- expand.grid(x = x_seq, y = y_seq)
ackley_function <- function(x, y, a = 20, b = 0.2, c = 2*pi) {
  -a * exp(-b * sqrt(0.5*(x^2 + y^2))) - exp(0.5*(cos(c*x) + cos(c*y))) + a + exp(1)
}
grid$z <- ackley_function(grid$x, grid$y)

# Global minimum for Ackley at (0,0)
global_minima <- data.frame(x = 0, y = 0)

p <- ggplot() +
  geom_raster(data = grid, aes(x = x, y = y, fill = z)) +
  scale_fill_viridis(option = "plasma", name = "Value") +
  geom_segment(data = segments_data,
               aes(x = X_0, y = X_1, xend = Xend, yend = Yend, group = OptIndex),
               arrow = arrow(length = unit(0.1, "cm")), color = "blue", alpha = 0.7) +
  geom_point(data = start_points, aes(x = X_0, y = X_1),
             color = "red", size = 1.5, alpha = 0.9) +
  geom_point(data = intermediate_points, aes(x = X_0, y = X_1),
             color = "deepskyblue", size = 1, alpha = 0.7) +
  geom_point(data = end_points, aes(x = X_0, y = X_1),
             color = "green", size = 1.5, alpha = 0.9) +
  geom_point(data = global_minima, aes(x = x, y = y),
             color = "red", size = 3, shape = 19) +
  labs(title = "Optimization Trajectories on Ackley Function",
       x = "X-axis", y = "Y-axis") +
  theme_minimal() +
  coord_fixed() +
  theme(legend.position = "right")

print(p)
ggsave("optimization_trajectories_ackley.pdf", plot = p, width = 8, height = 6, dpi = 300, device = cairo_pdf)
cat("Plot saved to optimization_trajectories_ackley.pdf\n")

