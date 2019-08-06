import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

sprite_map = cv2.imread('sequential_social_dilemma_games/spritemap-384.png')
#25 pixels x 25 pixels
#cherry = sprite_map[119:144, 0:25]
#ghost = sprite_map[143:168, 23:48] #red
#ghost2 = sprite_map[192:216, 0:24] #pink
#orange = sprite_map[119:144, 48:73]
#grape = sprite_map[119:144, 119:144]

#24 pixels x 24 pixels
cherry = sprite_map[120:144, 0:24]
ghost = sprite_map[144:168, 24:48] #red
ghost2 = sprite_map[192:216, 0:24] #pink
orange = sprite_map[120:144, 48:72]
grape = sprite_map[120:144, 120:144]


def save_img(rgb_arr, path, name):
    plt.imshow(rgb_arr, interpolation='nearest')
    plt.savefig(path + name)


def make_video_from_image_dir(vid_path, img_folder, video_name='trajectory', fps=5):
    """
    Create a video from a directory of images
    """
    images = [img for img in os.listdir(img_folder) if img.endswith(".png")]
    images.sort()

    rgb_imgs = []
    for i, image in enumerate(images):
        img = cv2.imread(os.path.join(img_folder, image))
        rgb_imgs.append(img)

    make_video_from_rgb_imgs(rgb_imgs, vid_path, video_name=video_name, fps=fps)


def make_video_from_rgb_imgs(rgb_arrs, vid_path, video_name='trajectory',
                             fps=5, format="mp4v", resize=(432, 600)):
    """
    Create a video from a list of rgb arrays
    """
    print("Rendering video...")
    if vid_path[-1] != '/':
        vid_path += '/'
    video_path = vid_path + video_name + '.mp4'

    if resize is not None:
        width, height = resize
    else:
        frame = rgb_arrs[0]
        height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*format)
    video = cv2.VideoWriter(video_path, fourcc, float(fps), (width, height))
    for i, image in enumerate(rgb_arrs):
        percent_done = int((i / len(rgb_arrs)) * 100)
        if percent_done % 20 == 0:
            print("\t...", percent_done, "% of frames rendered")

        if resize is not None:
            image = cv2.resize(image, resize, interpolation=cv2.INTER_NEAREST)
        image = change_to_sprite(image)
        video.write(image)

    video.release()
    cv2.destroyAllWindows()

def change_to_sprite(image):
    new_image=np.zeros([600,432,3], dtype=np.uint8)
    for row_idx in range(int(len(image)/24)):
        for col_idx in range(int(len(image[0])/24)):
            row=row_idx*24+5
            col=col_idx*24+5
            pixel = image[row][col]
            # if purple, change to ghost
            if pixel[2]==159 and pixel[1]==67:
                for pixel_row in range(len(ghost)):
                    for pixel_col in range(len(ghost[0])):
                        new_image[row-5 +pixel_row][col-5 +pixel_col] = ghost[pixel_row][pixel_col]
            # if yellow, change to ghost2
            elif pixel[1] == 223 and pixel[0]==16 and pixel[2]==238:
                for pixel_row in range(len(ghost2)):
                    for pixel_col in range(len(ghost2[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = ghost2[pixel_row][pixel_col]
            # if green, change to grapes
            elif pixel[1]==255 and pixel[0]==0 and pixel[2]==0:
                for pixel_row in range(len(grape)):
                    for pixel_col in range(len(grape[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = grape[pixel_row][pixel_col]
            #if blue, change to orange
            elif pixel[1]==81 and pixel[0]==154 and pixel[2]==5:
                for pixel_row in range(len(orange)):
                    for pixel_col in range(len(orange[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = orange[pixel_row][pixel_col]
            #if red, change to cherry
            elif pixel[2] == 245 and pixel[1]==0 and pixel[0]==100:
                for pixel_row in range(len(cherry)):
                    for pixel_col in range(len(cherry[0])):
                        new_image[row-5 + pixel_row][col-5 + pixel_col] = cherry[pixel_row][pixel_col]
    return new_image


def return_view(grid, pos, row_size, col_size):
    """Given a map grid, position and view window, returns correct map part

    Note, if the agent asks for a view that exceeds the map bounds,
    it is padded with zeros

    Parameters
    ----------
    grid: 2D array
        map array containing characters representing
    pos: list
        list consisting of row and column at which to search
    row_size: int
        how far the view should look in the row dimension
    col_size: int
        how far the view should look in the col dimension

    Returns
    -------
    view: (np.ndarray) - a slice of the map for the agent to see
    """
    x, y = pos
    left_edge = x - col_size
    right_edge = x + col_size
    top_edge = y - row_size
    bot_edge = y + row_size
    pad_mat, left_pad, top_pad = pad_if_needed(left_edge, right_edge,
                                               top_edge, bot_edge, grid)
    x += left_pad
    y += top_pad
    view = pad_mat[x - col_size: x + col_size + 1,
                   y - row_size: y + row_size + 1]
    return view


def pad_if_needed(left_edge, right_edge, top_edge, bot_edge, matrix):
    # FIXME(ev) something is broken here, I think x and y are flipped
    row_dim = matrix.shape[0]
    col_dim = matrix.shape[1]
    left_pad, right_pad, top_pad, bot_pad = 0, 0, 0, 0
    if left_edge < 0:
        left_pad = abs(left_edge)
    if right_edge > row_dim - 1:
        right_pad = right_edge - (row_dim - 1)
    if top_edge < 0:
        top_pad = abs(top_edge)
    if bot_edge > col_dim - 1:
        bot_pad = bot_edge - (col_dim - 1)

    return pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, 0), left_pad, top_pad


def pad_matrix(left_pad, right_pad, top_pad, bot_pad, matrix, const_val=1):
    pad_mat = np.pad(matrix, ((left_pad, right_pad), (top_pad, bot_pad)),
                     'constant', constant_values=(const_val, const_val))
    return pad_mat
