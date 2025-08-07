import os
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from PIL import Image

def debug_env(env):
    print('Env Info:')
    pprint(vars(env.env.env))
    print('\n\n')
    print('Setting Info')
    pprint(vars(env.env.env.setting))
    print('\n\n')
    print('Agent Info')
    pprint(vars(env.env.env.agent))
    print('\n\n')
    try:
        print('Play-style Wrapper Info')
        pprint(vars(env.env.env.agent.play_styles))
        print('\n\n')
    except:
        pass

def convert_fig(plt):
    # Get the figure and its axes
    fig = plt.gcf()
    axes = plt.gca()

    # Draw the content
    fig.canvas.draw()

    # Get the RGB values
    rgb = fig.canvas.tostring_rgb()

    # Get the width and height of the figure
    width, height = fig.canvas.get_width_height()

    # Convert the RGB values to a PIL Image
    img = Image.frombytes('RGB', (width, height), rgb)
    img_array = np.array(img)
    return img_array