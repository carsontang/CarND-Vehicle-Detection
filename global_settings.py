cell_per_block = 2
color_space = 'RGB'
heatmap_threshold = 30
hist_bins = 16
hog_channel = 'ALL' # 0, 1, 2, or "ALL"
orient = 30
pct_overlap = 0.75
pix_per_cell = 16
ring_buffer_cap = 30
spatial_size = (16, 16)

# Toggle extracting features from an image
use_hist_feat = False
use_hog_feat = True
use_spatial_feat = False

use_pretrained_model = False

x_start_stop = [0, 1280]
y_start_stop = [400, 720]