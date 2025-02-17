DOMAIN = "spaghetti_detection"
BRAND = "Spaghetti Detection"
DEFAULT_NAME = "Spaghetti Detection"

# Definitely not failing if ewm mean is below this level. =(0.4 - 0.02): 0.4 - optimal THRESHOLD_LOW in hyper params grid search; 0.02 - average of rolling_mean_short
THRESHOLD_LOW = 0.38
# Definitely failing if ewm mean is above this level. =(0.8 - 0.02): 0.8 - optimal THRESHOLD_HIGH in hyper params grid search; 0.02 - average of rolling_mean_short
THRESHOLD_HIGH = 0.78
# The number of frames at the beginning of the print that are considered "safe"
INIT_SAFE_FRAME_NUM = 30
# Print is failing is ewm mean is this many times over the short rolling mean
ROLLING_MEAN_SHORT_MULTIPLE = 3.8
# The multiplication factor to escalate "warning" to "error"
ESCALATING_FACTOR = 1.75

MAX_FRAME_NUM = 750
