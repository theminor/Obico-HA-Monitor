import logging
from .const import THRESHOLD_LOW, THRESHOLD_HIGH, INIT_SAFE_FRAME_NUM, ROLLING_MEAN_SHORT_MULTIPLE

LOGGER = logging.getLogger(__name__)

EWM_ALPHA = 2/(12 + 1)   # 12 is the optimal EWM span in hyper parameter grid search
ROLLING_WIN_SHORT = 310 # rolling window of 310 samples.
ROLLING_WIN_LONG = 7200 # rolling window of 7200 samples (~20 hours). Approximation of printer's base noise level

VISUALIZATION_THRESH = 0.2  # The thresh for a box to be drawn on the detective view

def update_prediction_with_detections(prediction, detections):

    p = sum_p_in_detections(detections)
    prediction.current_p = p
    prediction.current_frame_num += 1
    prediction.lifetime_frame_num += 1
    prediction.ewm_mean = next_ewm_mean(p, prediction.ewm_mean)
    prediction.rolling_mean_short = next_rolling_mean(p, prediction.rolling_mean_short, prediction.current_frame_num, ROLLING_WIN_SHORT)
    prediction.rolling_mean_long = next_rolling_mean(p, prediction.rolling_mean_long, prediction.lifetime_frame_num, ROLLING_WIN_LONG)

def is_failing(prediction, detective_sensitivity, escalating_factor=1):
    if prediction.current_frame_num < INIT_SAFE_FRAME_NUM:
        return False

    adjusted_ewm_mean = (prediction.ewm_mean - prediction.rolling_mean_long) * detective_sensitivity / escalating_factor
    if adjusted_ewm_mean < THRESHOLD_LOW:
        return False

    if adjusted_ewm_mean > THRESHOLD_HIGH:
        return True

    if adjusted_ewm_mean > (prediction.rolling_mean_short - prediction.rolling_mean_long) * ROLLING_MEAN_SHORT_MULTIPLE:
        return True

def next_ewm_mean(p, current_ewm_mean):
    return p * EWM_ALPHA + current_ewm_mean * (1-EWM_ALPHA)

# Approximation of rolling mean. inspired by https://dev.to/nestedsoftware/calculating-a-moving-average-on-streaming-data-5a7k
def next_rolling_mean(p, current_rolling_mean, count, win_size):
    return current_rolling_mean + (p - current_rolling_mean )/float(win_size if win_size <= count else count+1)

def sum_p_in_detections(detections):
    return sum([ d[1] for d in detections ])


class PrinterPrediction:
    def __init__(self):
        self.printer = None             # printer: A one-to-one relationship with the Printer model.
        self.current_frame_num = 0      # current_frame_num: The current frame number being processed.
        self.lifetime_frame_num = 0     # lifetime_frame_num: The total number of frames processed over the printer's lifetime.
        self.current_p = 0              # current_p: The current prediction score.
        self.ewm_mean = 0               # ewm_mean: The exponentially weighted mean of prediction scores.
        self.rolling_mean_short = 0     # rolling_mean_short: The short-term rolling mean of prediction scores.
        self.rolling_mean_long = 0      # rolling_mean_long: The long-term rolling mean of prediction scores
        self.created_at = None          # created_at: Timestamp when the prediction was created.
        self.updated_at = None          # updated_at: Timestamp when the prediction was last updated.
        
    def reset_for_new_print(self):
        self.current_frame_num = 0
        self.current_p = 0.0
        self.ewm_mean = 0.0
        self.rolling_mean_short = 0.0
        self.save()

    def __str__(self):
        return '| printer_id: {} | current_p: {:.4f} | ewm_mean: {:.4f} | rolling_mean_short: {:.4f} | rolling_mean_long: {:.4f} | current_frame_num: {} | lifetime_frame_num: {} |'.format(
            self.printer_id,
            self.current_p,
            self.ewm_mean,
            self.rolling_mean_short,
            self.rolling_mean_long,
            self.current_frame_num,
            self.lifetime_frame_num,
        )