from image_processing.Histogram import Histogram
from image_processing.PixelScaler import PixelScaler

dirs = ['../dataset/single-coins/train', '../dataset/single-coins/validation', '../dataset/single-coins/test']

histogram = Histogram(dirs, lower_threshold=0.05, upper_threshold=0.95)
histogram.count_histogram()
histogram.save_results()

scaler = PixelScaler(dirs, histogram.lower, histogram.upper)
scaler.scale_images_in_dirs()
