"""
Offering a simple sgf_from_image() conversion function between image and sgf format.

Heavily modified from:
# https://github.com/hanysz/img2sgf
# Alexander Hanysz, March 2020

- got rid of all the GUI stuff
"""

import math
import string
from bisect import bisect_left
from enum import Enum, IntEnum
from pathlib import Path
from tkinter import scrolledtext as scrolledtext

import cv2 as cv
import numpy as np
from PIL import Image, ImageEnhance
from sklearn.cluster import AgglomerativeClustering

HERE = Path(__file__).parent

BOARD_SIZE = 19
threshold_default = 250  # line detection votes threshold
black_stone_threshold_default = 128  # brightness on a scale of 0-255
black_stone_threshold = black_stone_threshold_default
edge_min_default = 50  # edge detection min threshold
edge_max_default = 200
sobel_default = 3  # edge detection: Sobel filter size, choose from 3, 5 or 7
gradient_default = 1  # edge detection: 1=L1 norm, 2=L2 norm
maxblur = 3  # make four blurred images (blur=1, 3, 5, 7) for circle detection
angle_tolerance = 1.0  # accept lines up to 1 degree away from horizontal or vertical
angle_delta = math.pi / 180 * angle_tolerance
min_grid_spacing = 10
big_space_ratio = 1.6  # try to fill in grid spaces that are >1.6 * min spacing
contrast_default = 50  # by default, raise the contrast a bit, it often seems to help!
brightness_default = 50  # don't change brightness


class Direction(Enum):
    HORIZONTAL = 1
    HORIZ = 1
    H = 1
    VERTICAL = 2
    VERT = 2


class BoardStates(IntEnum):
    EMPTY, BLACK, WHITE, STONE = range(4)
    # use STONE as temporary flag for colour not yet determined


class Alignment(IntEnum):
    TOP, BOTTOM, LEFT, RIGHT = range(4)


valid_grid = False

selection_global = np.array((0, 0, 0, 0))  # current region relative to original image

stone_brightnesses = []


# Part 2: image processing functions


def rectangle_centre(a):
    return np.array(((a[0] + a[2]) / 2, a[1] + a[3] / 2))


def crop_and_rotate_image():
    global region_PIL
    rotation_centre = tuple(rectangle_centre(selection_global))
    region_PIL = input_image_PIL.rotate(
        angle=0, fillcolor="white", center=rotation_centre
    ).crop(selection_global)


def process_image():
    global input_image_np, edge_detected_image_np, circles, circles_removed_image_np, grey_image_np, region_PIL
    # photos (_PIL images) need to be global so that the garbage collector doesn't
    # clean them up and blank out the canvases
    # numpy images (_np) are used by other functions
    global threshold_hist, threshold_line
    # global so that other functions can move and redraw the line
    global valid_grid
    # keep other functions informed of processing status

    valid_grid = False

    log("\nProcessing image")
    crop_and_rotate_image()

    log("Contrast = " + str(contrast_default))
    scaled_contrast = 102 / (101 - contrast_default) - 1
    # convert range 0-100 into range 0.01-101, with 50->1.0
    region_PIL = ImageEnhance.Contrast(region_PIL).enhance(scaled_contrast)

    log("Brightness = " + str(brightness_default))
    scaled_brightness = 450 / (200 - brightness_default) - 2
    # convert range 0-100 into range 0.25-2.5, with 50->1.0
    region_PIL = ImageEnhance.Brightness(region_PIL).enhance(scaled_brightness)
    input_image_np = np.array(region_PIL)

    log("Converting to greyscale")
    grey_image_np = cv.cvtColor(input_image_np, cv.COLOR_BGR2GRAY)

    log("Running Canny edge detection algorithm")
    # no point logging the parameters now I've turned off the UI for changing them
    edge_detected_image_np = cv.Canny(
        input_image_np,
        edge_min_default,
        edge_max_default,
        apertureSize=sobel_default,
        L2gradient=(gradient_default == 2),
    )

    log("Detecting circles")
    circles_removed_image_np = edge_detected_image_np.copy()
    # Make a few different blurred versions of the image, so we can find most of the circles
    blurs = [grey_image_np, edge_detected_image_np]
    for i in range(maxblur + 1):
        b = 2 * i + 1
        blurs.append(cv.medianBlur(grey_image_np, b))
        blurs.append(cv.GaussianBlur(grey_image_np, (b, b), b))

    first_circles = True
    circles = []
    for b in blurs:
        c = cv.HoughCircles(b, cv.HOUGH_GRADIENT, 1, 10, np.array([]), 100, 30, 1, 30)
        if len(c) > 0:
            if first_circles:
                circles = c[0]
                first_circles = False
            else:
                circles = np.vstack((circles, c[0]))

    # For each circle, erase the bounding box and replace by a single pixel in the middle
    # This makes it easier to detect grid lines when
    # there are lots of stones on top of the line
    for i in range(len(circles)):
        xc, yc, r = circles[i, :]
        r = (
            r + 2
        )  # need +2 because circle edges can stick out a little past the bounding box
        ul = (int(round(xc - r)), int(round(yc - r)))
        lr = (int(round(xc + r)), int(round(yc + r)))
        middle = (int(round(xc)), int(round(yc)))
        cv.rectangle(circles_removed_image_np, ul, lr, (0, 0, 0), -1)  # -1 = filled
        cv.circle(circles_removed_image_np, middle, 1, (255, 255, 255), -1)

    find_grid()


def find_lines(threshold, direction):
    # Lines are assumed to be horizontal or vertical
    # Return value is a vector of x- or y-intercepts
    # Remember that horizontal lines intercept the y-axis,
    #   be careful not to get x and y the wrong way round!
    if direction == Direction.H:
        lines = cv.HoughLines(
            circles_removed_image_np,
            rho=1,
            theta=math.pi / 180.0,
            threshold=threshold,
            min_theta=math.pi / 2 - angle_delta,
            max_theta=math.pi / 2 + angle_delta,
        )
    else:
        vlines1 = cv.HoughLines(
            circles_removed_image_np,
            rho=1,
            theta=math.pi / 180.0,
            threshold=threshold,
            min_theta=0,
            max_theta=angle_delta,
        )
        vlines2 = cv.HoughLines(
            circles_removed_image_np,
            rho=1,
            theta=math.pi / 180.0,
            threshold=threshold,
            min_theta=math.pi - angle_delta,
            max_theta=math.pi,
        )
        if vlines2 is not None:
            vlines2[:, 0, 0] = -vlines2[:, 0, 0]
            vlines2[:, 0, 1] = vlines2[:, 0, 1] - math.pi
            if vlines1 is not None:
                lines = np.vstack((vlines1, vlines2))
            else:
                lines = vlines2
        else:
            lines = vlines1
    return [] if lines is None else lines[:, 0, 0].reshape(-1, 1)
    # reshape because clustering function prefers column vector not row


def find_all_lines():
    hlines = find_lines(threshold_default, Direction.HORIZONTAL)
    hcount = len(hlines)
    vlines = find_lines(threshold_default, Direction.VERTICAL)
    vcount = len(vlines)
    log(
        "Found "
        + str(hcount)
        + " distinct horizontal lines and "
        + str(vcount)
        + " distinct vertical lines"
    )
    return (hlines, vlines)


def find_clusters_fixed_threshold(threshold, direction):
    lines = find_lines(threshold, direction)
    if lines is not None:
        cluster_model = AgglomerativeClustering(
            n_clusters=None, linkage="single", distance_threshold=min_grid_spacing
        )
        try:
            answer = cluster_model.fit(lines)
            # this may fail if there's not enough lines
            return answer
        except:
            return None
    else:
        return None


def get_cluster_centres(model, points):
    if model is None:
        return []
    n = model.n_clusters_
    answer = np.zeros(n)
    for i in range(n):
        this_cluster = points[model.labels_ == i]
        answer[i] = this_cluster.mean()
    answer.sort()
    return answer


def cluster_lines(hlines, vlines):
    hclusters = find_clusters_fixed_threshold(threshold_default, Direction.HORIZ)
    hcentres = get_cluster_centres(hclusters, hlines)
    vsize_initial = len(hcentres) if hcentres is not None else 0
    vclusters = find_clusters_fixed_threshold(threshold_default, Direction.VERT)
    vcentres = get_cluster_centres(vclusters, vlines)
    hsize_initial = len(vcentres) if vcentres is not None else 0

    log(
        "Got "
        + str(vsize_initial)
        + " horizontal and "
        + str(hsize_initial)
        + " vertical grid lines"
    )

    return (hcentres, vcentres)


def complete_grid(x):
    # Input: x is a set of grid coordinates, possibly with gaps
    #   stored as a numpy row vector, sorted
    # Output: x with gaps filled in, if that's plausible, otherwise None if grid is invalid
    if x is None or len(x) == 0:
        log("No grid lines found at all!")
        return None

    if len(x) == 1:
        log("Only found one grid line")
        return None

    spaces = x[1:] - x[:-1]
    # Some of the spaces should equal the grid spacing, while some will be bigger because   of gaps
    min_space = min(spaces)
    if min_space < min_grid_spacing:
        log(
            "Grid lines are too close together: minimum spacing is "
            + str(min_space)
            + "     pixels"
        )
        return None
    bound = min_space * big_space_ratio
    big_spaces = spaces[spaces > bound]
    if len(big_spaces) == 0:  # no gaps!
        log("Got a complete grid of " + str(len(x)) + " lines")
        return x
    small_spaces = spaces[spaces <= bound]
    max_space = max(small_spaces)
    average_space = (min_space + max_space) / 2

    # Calculate total grid size, and check for weird gaps along the way
    n = len(small_spaces)
    for s in big_spaces:
        m = s / average_space
        n += int(round(m))
    if n > BOARD_SIZE + 2:
        log("Distance between edges of grid is " + str(n) + " times minimum space.")
        log("Extra lines on diagram, or a grid line detected twice?")
        return None

    # Now we know we have a valid grid (except maybe too big).  Let's fill in the gaps.
    n += 1  # need to increment because one gap equals two grid lines, two gaps=three lines  etc
    log("Got " + str(len(x)) + " lines within a grid of size " + str(n))
    if len(x) < n:
        log("Filling in gaps.")
        answer = np.zeros(n)
        answer[0] = x[0]
        i, j = 1, 1  # i points to answer grid, j points to x grid
        for s in spaces:
            if s <= max_space:
                answer[i] = x[j]
                i += 1
                j += 1
            else:
                m = int(round(s / average_space))
                for k in range(m):
                    answer[i] = (
                        x[j - 1] + (k + 1) * s / m
                    )  # linearly interpolate the missing 'x's
                    i += 1
                j += 1  # yes, that's right, we've incremented i 'm' times but j only once
        return answer
    else:
        return x


def truncate_grid(x):
    # x is a vector of grid coordinates as for complete_grid()
    # if size of x exceed board size by 1 or 2,
    # the extra lines are likely to be a bounding box, board edge or text
    # so we should drop them
    if x is None:
        return None
    if len(x) == BOARD_SIZE + 2:
        # two extra lines are likely to be a bounding box or board edges in the image
        # so let's drop them
        log("Dropping two extra lines at the outsides of the grid")
        return x[1:-1]
    if len(x) == BOARD_SIZE + 1:
        # most likely scenario is horizontal lines with a diagram caption underneath,
        # and the text is recognised as an extra line
        log("Dropping one extra line at the end of the grid")
        return x[:-1]
    return x


def validate_grid(hcentres, vcentres):
    log("Assessing horizontal lines.")
    hcentres = truncate_grid(hcentres)
    hcentres_complete = complete_grid(hcentres)
    hcentres_complete = truncate_grid(hcentres_complete)
    if hcentres_complete is None:
        return [False, circles, 0, 0] + 4 * [None]
    log("Assessing vertical lines.")
    vcentres = truncate_grid(vcentres)
    vcentres_complete = complete_grid(vcentres)
    vcentres_complete = truncate_grid(vcentres_complete)
    if vcentres_complete is None:
        return [False, circles, 0, 0] + 4 * [None]

    # Later we'll need the grid size and average spacing
    vsize, hsize = len(hcentres_complete), len(vcentres_complete)
    # Note: number of *horizontal* lines is the *vertical* sides of the board!
    hspace = (hcentres_complete[-1] - hcentres_complete[0]) / vsize
    vspace = (vcentres_complete[-1] - vcentres_complete[0]) / hsize
    # And now that we know the spacing, let's get rid of any circles that are the wrong size
    # (sometimes you get little circles from bits of letters and numbers on the diagram)
    min_circle_size = (
        min(hspace, vspace) * 0.3
    )  # diameter must be > 60% of grid spacing
    max_circle_size = max(hspace, vspace) * 0.65  # and less than 130% of grid spacing
    newcircles = [c for c in circles if min_circle_size < c[2] < max_circle_size]
    return (
        True,
        newcircles,
        vsize,
        hsize,
        hcentres_complete,
        vcentres_complete,
        hspace,
        vspace,
    )


def closest_index(a, x):
    # Input: a is a number, x a sorted list of numbers
    # Output: the index of the element of x closest to a
    # Break ties to the left (smaller index)
    i = bisect_left(x, a)
    # This is the index of the largest element of x that's smaller than a
    if i == 0:
        return 0
    if i == len(x):
        return i - 1
    # else i is between 1 and len(x)-1 inclusive
    return i - 1 if a - x[i - 1] <= x[i] - a else i


def closest_grid_index(p):
    # Input: p is (x, y) coordinates of a pixel (usually a circle centre)
    # Output: (i, j) coordinates of p on the board
    return (
        closest_index(p[0], vcentres_complete),
        closest_index(p[1], hcentres_complete),
    )


def average_intensity(i, j):
    # Input: i, j are grid coordinates of a point on the board
    # Output: average pixel intensity of a neighbourhood of p,
    # to help distinguish between black and white stones
    x = vcentres_complete[i]
    xmin, xmax = int(round(x - hspace / 2)), int(round(x + hspace / 2))
    y = hcentres_complete[j]
    ymin, ymax = int(round(y - vspace / 2)), int(round(y + vspace / 2))
    # Truncate coordinates to stay in bounds: sometimes circles can go outside the image
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(grey_image_np.shape[1], xmax)
    ymax = min(grey_image_np.shape[0], ymax)
    return np.mean(grey_image_np[ymin:ymax, xmin:xmax])  # nb flip x,y for np indexing


def align_board(b, a):
    # b is a part board, a is an alignment (top, left, etc)
    # return a full board with b in the appropriate side/quadrant
    board = np.zeros((BOARD_SIZE, BOARD_SIZE))

    xoffset = BOARD_SIZE - hsize if a[0] == Alignment.RIGHT else 0
    yoffset = BOARD_SIZE - vsize if a[1] == Alignment.BOTTOM else 0
    for i in range(hsize):
        for j in range(vsize):
            board[i + xoffset, j + yoffset] = b[i, j]
    return board


def identify_board():
    global detected_board, full_board, stone_brightnesses, num_black_stones, num_white_stones

    log("Guessing stone colours based on a threshold of " + str(black_stone_threshold))
    detected_board = np.zeros((hsize, vsize))
    num_black_stones, num_white_stones = 0, 0
    for c in circles:
        detected_board[closest_grid_index(c[0:2])] = BoardStates.STONE

    num_stones = np.count_nonzero(detected_board)
    stone_brightnesses = np.zeros(num_stones)
    i = 0
    for j in range(hsize):
        for k in range(vsize):
            if detected_board[j, k] == BoardStates.STONE:
                stone_brightnesses[i] = average_intensity(j, k)
                i += 1
    num_black_stones = sum(stone_brightnesses <= black_stone_threshold)
    black_text = str(num_black_stones) + " black stone"
    if num_black_stones != 1:
        black_text += "s"
    num_white_stones = num_stones - num_black_stones
    white_text = str(num_white_stones) + " white stone"
    if num_white_stones != 1:
        white_text += "s"
    log(
        "Detected "
        + black_text
        + " and "
        + white_text
        + " on a "
        + str(hsize)
        + "x"
        + str(vsize)
        + " board."
    )

    for i in range(hsize):
        for j in range(vsize):
            if detected_board[i, j] == BoardStates.STONE:
                x = average_intensity(i, j)
                detected_board[i, j] = (
                    BoardStates.BLACK
                    if x <= black_stone_threshold
                    else BoardStates.WHITE
                )
    full_board = align_board(detected_board.copy(), board_alignment)


def find_grid():
    global valid_grid, circles, vsize, hsize, hspace, vspace, hcentres, vcentres, hcentres_complete, vcentres_complete

    hlines, vlines = find_all_lines()
    hcentres, vcentres = cluster_lines(hlines, vlines)
    (
        valid_grid,
        circles,
        vsize,
        hsize,
        hcentres_complete,
        vcentres_complete,
        hspace,
        vspace,
    ) = validate_grid(hcentres, vcentres)
    if valid_grid:
        if hsize > BOARD_SIZE:
            log("Too many vertical lines!")
        elif vsize > BOARD_SIZE:
            log("Too many horizontal lines!")
        else:
            identify_board()


def to_SGF(board, black_to_move=True):
    board_letters = string.ascii_lowercase  # 'a' to 'z'
    output = "(;GM[1]FF[4]SZ[" + str(BOARD_SIZE) + "]\n"
    if black_to_move:
        output += "PL[B]\n"
    else:
        output += "PL[W]\n"
    black_moves, white_moves = "", ""
    if BoardStates.BLACK in board:
        black_moves += "AB"
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == BoardStates.BLACK:
                    black_moves += "[" + board_letters[i] + board_letters[j] + "]"
    if BoardStates.WHITE in board:
        white_moves += "AW"
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board[i, j] == BoardStates.WHITE:
                    white_moves += "[" + board_letters[i] + board_letters[j] + "]"
    if black_to_move:
        output += black_moves + "\n" + white_moves + "\n" + ")\n"
    else:
        output += white_moves + "\n" + black_moves + "\n" + ")\n"
    return output


def save_SGF(file: str | Path):
    sgf = to_SGF(full_board)
    with open(file, "w") as f:
        f.write(sgf)


def log(text: str):
    print(text)


def initialise_parameters():
    global region_PIL, valid_grid, board_alignment, black_stone_threshold, selection_global

    valid_grid = False
    board_alignment = [Alignment.LEFT, Alignment.TOP]
    black_stone_threshold = black_stone_threshold_default

    region_PIL = input_image_PIL.copy()
    selection_global = np.array([0, 0] + list(region_PIL.size))
    process_image()


def sgf_from_image(image: str | Path) -> str:
    global input_image_PIL
    input_image_PIL = Image.open(image).convert("RGB")
    initialise_parameters()
    process_image()
    return to_SGF(full_board)


if __name__ == "__main__":
    # image = HERE / "plot_20241004_095831.png"
    image = HERE / "screen.png"
    sgf = sgf_from_image(image)
    print("sgf", sgf)
    save_SGF(HERE / "output.sgf")
