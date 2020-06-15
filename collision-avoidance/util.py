import numpy as np
import PIL.Image as Image
import algorithm
import matplotlib.pyplot as plt

DEPTH_MIN, DEPTH_MAX = 1, 100


def visualize_depth(depth_data):
    clipped_data = np.clip(depth_data, DEPTH_MIN, DEPTH_MAX)
    transformed_data = (
        DEPTH_MAX
        * (np.log2(clipped_data) - np.log2(DEPTH_MIN))
        / (np.log2(DEPTH_MAX) - np.log2(DEPTH_MIN))
    )
    normalized_data = transformed_data * (255 / DEPTH_MAX)
    inversed_data = 255 - normalized_data.astype(np.uint8)
    return inversed_data


def save_algorithm(depth_data, algo_result, path):
    image = Image.fromarray(visualize_depth(depth_data), mode="L").convert("RGB")
    for i in range(depth_data.shape[0]):
        for j in range(depth_data.shape[1]):
            if depth_data[i, j] < algorithm.SAFE_DISTANCE:
                image.putpixel((j, i), (255, 0, 0))
    #
    plt.clf()
    plt.axis("off")
    plt.imshow(image)
    arrow_size = 50
    mark_pos = (image.width / 2, image.height / 2)
    if algo_result == algorithm.Result.STOP:
        plt.plot(*mark_pos, color="b", marker="X", markersize=30)
    elif algo_result == algorithm.Result.UP_LEFT:
        plt.arrow(*mark_pos, -arrow_size, -arrow_size, color="b", head_width=10)
    elif algo_result == algorithm.Result.UP:
        plt.arrow(*mark_pos, 0, -arrow_size, color="b", head_width=10)
    elif algo_result == algorithm.Result.UP_RIGHT:
        plt.arrow(*mark_pos, arrow_size, -arrow_size, color="b", head_width=10)
    elif algo_result == algorithm.Result.RIGHT:
        plt.arrow(*mark_pos, arrow_size, 0, color="b", head_width=10)
    elif algo_result == algorithm.Result.DOWN_RIGHT:
        plt.arrow(*mark_pos, arrow_size, arrow_size, color="b", head_width=10)
    elif algo_result == algorithm.Result.DOWN:
        plt.arrow(*mark_pos, 0, arrow_size, color="b", head_width=10)
    elif algo_result == algorithm.Result.DOWN_LEFT:
        plt.arrow(*mark_pos, -arrow_size, arrow_size, color="b", head_width=10)
    elif algo_result == algorithm.Result.LEFT:
        plt.arrow(*mark_pos, -arrow_size, 0, color="b", head_width=10)
    plt.savefig(path)
