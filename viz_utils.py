import numpy as np
def draw_box(canvas: np.ndarray, grid_position: tuple, grid_h: int, grid_w: int, value: float):
    _, pix_h, pix_w = tuple(canvas.shape)
    pix_h_per_box = pix_h / grid_h
    pix_w_per_box = pix_w / grid_w
    h_low = int(pix_h_per_box * grid_position[0])
    h_high = int(pix_h_per_box * (grid_position[0] + 1))
    w_low = int(pix_w_per_box * grid_position[1])
    w_high = int(pix_w_per_box * (grid_position[1] + 1))
    
    color = np.array([255 * min(1, 2 * (1 - value)), 255 * min(1, 2 * value), 0])
    # if h_low == 0:
    #     color = np.zeros(3)
    # elif w_low == 0:
    #     color = np.ones(3) * 255
    # print(w_low, w_high, h_low, h_high, color, value)

    canvas[:, h_low:h_high, w_low:w_high] = color[:, None, None]