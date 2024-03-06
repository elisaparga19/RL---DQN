import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
from torch.nn import functional as F
from typing import Iterable
from torch.nn import Module

# Plots min, max and mean + standard deviation bars of a population over time
def lineplot(xs, ys_population, title, path='', xaxis='episode'):
    max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

    if isinstance(ys_population[0], list) or isinstance(ys_population[0], tuple):
        ys = np.asarray(ys_population, dtype=np.float32)
        ys_min, ys_max, ys_mean, ys_std, ys_median = ys.min(1), ys.max(1), ys.mean(1), ys.std(1), np.median(ys, 1)
        ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

        trace_max = Scatter(x=xs, y=ys_max, line=Line(color=max_colour, dash='dash'), name='Max')
        trace_upper = Scatter(x=xs, y=ys_upper, line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
        trace_mean = Scatter(x=xs, y=ys_mean, fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
        trace_lower = Scatter(x=xs, y=ys_lower, fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
        trace_min = Scatter(x=xs, y=ys_min, line=Line(color=max_colour, dash='dash'), name='Min')
        trace_median = Scatter(x=xs, y=ys_median, line=Line(color=max_colour), name='Median')
        data = [trace_upper, trace_mean, trace_lower, trace_min, trace_max, trace_median]
    else:
        data = [Scatter(x=xs, y=ys_population, line=Line(color=mean_colour))]
    plotly.offline.plot({
        'data': data,
        'layout': dict(title=title, xaxis={'title': xaxis}, yaxis={'title': title})
    }, filename=os.path.join(path, title + '.html'), auto_open=False)