import os
from visdom import Visdom
import numpy as np

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', filepath='mse_runs/log.log'):
        self.viz = Visdom(port=6006, log_to_filename=filepath)
        if os.path.exists(filepath):
            self.viz.replay_log(filepath)
        self.env = env_name
        self.plots = {}
        self.images = {}

    def plot(self, legend_name, title_name, x, y):
        if title_name not in self.plots:
            self.plots[title_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]),
                                                env=self.env, 
                                                win=title_name,
                                                opts=dict(
                                                    legend=[legend_name],
                                                    title=title_name,
                                                    xlabel='Epochs',
                                                    ylabel='Loss value'
                                                    ),
                                                )
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env,
                            win=title_name, name=legend_name,
                            update = 'append')

    def draw_image(self, image, win_name = 'An_image', caption = 'An image'):
        self.viz.image(image,
                    env = self.env,
                    win = win_name,
                    opts = dict(
                        caption = caption,
                        width = 500,
                        height = 500,
                        store_history = True
                        )
                    )

    def draw_3_images(self, images, win_name = '3_images', caption = 'Input, GT, Output'):
        self.viz.images(images,
                    env = self.env,
                    win = win_name,
                    opts = dict(
                        nrow = 3,
                        padding = 5,
                        caption = caption,
                        width = 1400,
                        height = 400,
                        store_history = True,
                        )
                    )

