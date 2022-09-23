import os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from imageio import imwrite
from data_reader import read_gaze_data_csv_file as read_gaze

all_gaze_files = Path.cwd().rglob('./asterix/*.txt')

def check_gaze_range(pos_x, pos_y, w=160, h=210):
    """Helper function to check whether the gaze
    coordinate is within the screen area"""
    if pos_x < 0 or pos_x > w or pos_y < 0 or pos_y > h:
        return False
    return True

for file in all_gaze_files:
    gaze_positions, _, _, _, _, _, frameid_list = read_gaze(file)
    for frame_id in frameid_list:
        gaze_list = gaze_positions[frame_id]
        heatmap = np.zeros((210, 160))
        if gaze_list is not None and len(gaze_list) > 0:
            for (x, y) in gaze_list:
                if check_gaze_range(x, y):
                    heatmap[int(y - 1), int(x - 1)] += 1

        # sigma should be one visual degree
        # 0.40 visual degrees = 2.94 pixels (Zhang et al. 2019, p. 3)
        # 1 visual degree = 7.35 pixels or, rounded, 7 pixels
        heatmap = ndi.gaussian_filter(heatmap, sigma=7)
        heatmap = np.interp(heatmap, (heatmap.min(), heatmap.max()), (0, 255)) # change pixel values range from [0,1] to [0, 255]
        imwrite(f"./asterix/heatmaps/{frame_id}.png", heatmap.astype(np.uint8))

heatmaps_list = os.listdir("./asterix/heatmaps/")
frames_list = os.listdir("./asterix/frames/")
assert len(heatmaps_list) == len(frames_list), "Number of heatmaps is not the same as the number of frames"
