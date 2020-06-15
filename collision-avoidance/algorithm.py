import enum
import numpy as np

IMAGE_WIDTH, IMAGE_HEIGHT = 160, 192
FLIGHT_WIDTH, FLIGHT_HEIGHT = 2, 1
SAFE_DISTANCE = 10
DANGER_PIXEL_RATIO = 0.0075
DANGER_WEIGHT_SUM = 5


class Result(enum.Enum):
    SAFE = 0
    LEFT = 1
    UP_LEFT = 2
    UP = 3
    UP_RIGHT = 4
    RIGHT = 5
    DOWN_RIGHT = 6
    DOWN = 7
    DOWN_LEFT = 8
    STOP = 9


def depth_to_3d(depth_image):
    points_3d = np.empty(depth_image.shape, dtype=tuple)
    for i in range(len(depth_image)):
        for j in range(len(depth_image[i])):
            z = depth_image[i, j]
            y = (
                depth_image[i, j]
                * (i - (IMAGE_HEIGHT - 1) / 2)
                / (IMAGE_HEIGHT / 2)
                * -1
            )
            x = depth_image[i, j] * (j - (IMAGE_WIDTH - 1) / 2) / (IMAGE_WIDTH / 2)
            points_3d[i, j] = (x, y, z)
    return points_3d


def path_safe(points_3d):
    danger_cnt = 0
    for i in range(points_3d.shape[0]):
        for j in range(points_3d.shape[1]):
            if (
                abs(points_3d[i, j][0]) <= FLIGHT_WIDTH
                and abs(points_3d[i, j][1]) <= FLIGHT_HEIGHT
                and points_3d[i, j][2] <= SAFE_DISTANCE
            ):
                danger_cnt += 1
    return danger_cnt < (IMAGE_WIDTH * IMAGE_HEIGHT * DANGER_PIXEL_RATIO)


def approach_safe(depth_image):
    points_3d = depth_to_3d(depth_image)
    #
    if path_safe(points_3d):
        return Result.SAFE
    #
    area_sum = np.zeros(8)
    for i in range(points_3d.shape[0]):
        for j in range(points_3d.shape[1]):
            if not points_3d[i, j][2] >= SAFE_DISTANCE:
                continue
            #
            area = points_3d[i, j][2] ** 2
            if abs(points_3d[i, j][0]) >= abs(points_3d[i, j][1]):
                if points_3d[i, j][0] >= 0:
                    if points_3d[i, j][1] >= 0:
                        area_sum[1] += area
                    else:
                        area_sum[2] += area
                else:
                    if points_3d[i, j][1] >= 0:
                        area_sum[6] += area
                    else:
                        area_sum[5] += area
            else:
                if points_3d[i, j][1] >= 0:
                    if points_3d[i, j][0] >= 0:
                        area_sum[0] += area
                    else:
                        area_sum[7] += area
                else:
                    if points_3d[i, j][0] >= 0:
                        area_sum[3] += area
                    else:
                        area_sum[4] += area
    #
    merged_sum = [area_sum[idx] + area_sum[(idx + 1) % 8] for idx in range(8)]
    result_dir = np.argmax(merged_sum)
    if result_dir == 0:
        return Result.UP_RIGHT
    elif result_dir == 1:
        return Result.RIGHT
    elif result_dir == 2:
        return Result.DOWN_RIGHT
    elif result_dir == 3:
        return Result.DOWN
    elif result_dir == 4:
        return Result.DOWN_LEFT
    elif result_dir == 5:
        return Result.LEFT
    elif result_dir == 6:
        return Result.UP_LEFT
    elif result_dir == 7:
        return Result.UP


def away_danger(depth_image):
    points_3d = depth_to_3d(depth_image)
    #
    if path_safe(points_3d):
        return Result.SAFE
    #
    weight_sum = np.zeros(8)
    for i in range(points_3d.shape[0]):
        for j in range(points_3d.shape[1]):
            if points_3d[i, j][2] >= SAFE_DISTANCE:
                continue
            #
            weight = points_3d[i, j][2] ** (1 / 2)
            if abs(points_3d[i, j][0]) >= abs(points_3d[i, j][1]):
                if points_3d[i, j][0] >= 0:
                    if points_3d[i, j][1] >= 0:
                        weight_sum[1] += weight
                    else:
                        weight_sum[2] += weight
                else:
                    if points_3d[i, j][1] >= 0:
                        weight_sum[6] += weight
                    else:
                        weight_sum[5] += weight
            else:
                if points_3d[i, j][1] >= 0:
                    if points_3d[i, j][0] >= 0:
                        weight_sum[0] += weight
                    else:
                        weight_sum[7] += weight
                else:
                    if points_3d[i, j][0] >= 0:
                        weight_sum[3] += weight
                    else:
                        weight_sum[4] += weight
    #
    merged_sum = [weight_sum[idx] + weight_sum[(idx + 1) % 8] for idx in range(8)]
    result_dir = np.argmax(merged_sum)
    if result_dir == 0:
        return Result.DOWN_LEFT
    elif result_dir == 1:
        return Result.LEFT
    elif result_dir == 2:
        return Result.UP_LEFT
    elif result_dir == 3:
        return Result.UP
    elif result_dir == 4:
        return Result.UP_RIGHT
    elif result_dir == 5:
        return Result.RIGHT
    elif result_dir == 6:
        return Result.DOWN_RIGHT
    elif result_dir == 7:
        return Result.DOWN


def two_step_decision(depth_image):
    points_3d = depth_to_3d(depth_image)
    #
    if path_safe(points_3d):
        return Result.SAFE
    #
    danger_sum = np.zeros(8)
    safe_sum = np.zeros(8)
    for i in range(points_3d.shape[0]):
        for j in range(points_3d.shape[1]):
            if points_3d[i, j][2] < SAFE_DISTANCE:
                target_sum = danger_sum
                weight = points_3d[i, j][2] ** -10
            else:
                target_sum = safe_sum
                weight = points_3d[i, j][2] ** 2
            #
            if abs(points_3d[i, j][0]) >= abs(points_3d[i, j][1]):
                if points_3d[i, j][0] >= 0:
                    if points_3d[i, j][1] >= 0:
                        target_sum[1] += weight
                    else:
                        target_sum[2] += weight
                else:
                    if points_3d[i, j][1] >= 0:
                        target_sum[6] += weight
                    else:
                        target_sum[5] += weight
            else:
                if points_3d[i, j][1] >= 0:
                    if points_3d[i, j][0] >= 0:
                        target_sum[0] += weight
                    else:
                        target_sum[7] += weight
                else:
                    if points_3d[i, j][0] >= 0:
                        target_sum[3] += weight
                    else:
                        target_sum[4] += weight
    #
    merged_danger = [danger_sum[idx] + danger_sum[(idx + 1) % 8] for idx in range(8)]
    merged_safe = [safe_sum[idx] + safe_sum[(idx + 1) % 8] for idx in range(8)]
    print("danger:", merged_danger)
    print("safe:", merged_safe)
    best_choice = None
    for idx in range(8):
        if merged_danger[idx] < DANGER_WEIGHT_SUM:
            if best_choice == None:
                best_choice = idx
            elif merged_safe[idx] > merged_safe[best_choice]:
                best_choice = idx
    #
    if best_choice == 0:
        return Result.UP_RIGHT
    elif best_choice == 1:
        return Result.RIGHT
    elif best_choice == 2:
        return Result.DOWN_RIGHT
    elif best_choice == 3:
        return Result.DOWN
    elif best_choice == 4:
        return Result.DOWN_LEFT
    elif best_choice == 5:
        return Result.LEFT
    elif best_choice == 6:
        return Result.UP_LEFT
    elif best_choice == 7:
        return Result.UP
    else:
        return Result.STOP

