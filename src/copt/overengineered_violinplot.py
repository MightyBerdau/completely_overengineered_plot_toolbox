import collections
import copy
import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.optimize import minimize

VALID_SNS_PALETTES =[
    'tab10', 'deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'
    ]
M_BOUNDS = (-1, +1) # bounds for formula for determining optimal scatter spread of data points in violin plots
EPS=1e-6 # avoiding numerical instabilities...

def negative_sum_distances_with_penalty(
        m:np.ndarray,
        y:np.ndarray,
        x_max:np.ndarray,
        spread_strength:float,
        spread_power:float) -> float:
    
    """Function to be minimized for finding optimal scattering of data points in violin

    Actually the sum of distances should be maximized, but since I could not
    find a function for maximizing I am using the negative sum and try to
    minimize it...

    To avoid all data points being spread to the bounds of the violin plot,
    a penalty term is used, which kind of works like Coulomb's law.
    Depending on the penalty arguments chosen, data points 'repel' each other.
    This forces the algorithm to spread data points somewhere inside the violin.

    Args:
        m (np.ndarray): Spread factor for each data point i
        y (np.ndarray): Measured value of each data point in violin plot
        x_max (np.ndarray): Maximum spread defined by violin plot shape
        spread_strength (float): The higher the value,
            the more data points are spread.
        spread_power (float): Punishes very small distances.

    Returns:
        float: Negative sum of distances of all data points to each other
    """
    # Calculating the scatter position y
    x = m * x_max  # with b = max distance to violin bounds and m to be found in range (-1, +1)

    # Expressing data points as 2D, where x = measurement data and y = scattering offset
    data_points = np.stack([x, y], axis=1)

    # Calculating distances of all data points to each other
    dists = data_points[:, None, :] - data_points[None, :, :]

    # Expressing sum of all distances (which is to be maximized)
    sum_of_dists = np.linalg.norm(dists, axis=-1)

    # dists is a matrix that contains each distance of point i and j twice...
    actual_sum_of_dists = np.sum(np.triu(sum_of_dists, 1))

    # Applying a penalty to avoid all data points to spread just to violin bounds
    penalty = np.sum(np.triu((spread_strength / (sum_of_dists + EPS)) ** spread_power, 1))

    return -actual_sum_of_dists + penalty # Returning negative sum to be minimized, since there is no maximize function, apparently...

def find_optimal_spread_params(
        group_data:np.ndarray,
        x_max:np.ndarray,
        spread_strength:float,
        spread_power:float,
        m_min:float,
        m_max:float):
    """Finds optimal scattering of data points by minimizing negative_sum_distances_with_penalties()

    Args:
        group_data (np.ndarray): Data as a 1D array
        x_max (np.ndarray): Maximum spread defined by violin plot shape
        spread_strength (float): The higher the value,
            the more data points are spread.
        spread_power (float): Punishes very small distances.
        m_min (float): Min value for m.
        m_max (float): Max value for m.

    Returns:
        _type_: Jitter (scattering offest on x-axis)
    """
    m_bounds = (m_min, m_max)
    m0 = np.linspace(m_bounds[0], m_bounds[1], len(group_data))
    bounds = [m_bounds] * len(group_data)

    res = minimize(
        negative_sum_distances_with_penalty,
        m0,
        args=(group_data, x_max, spread_strength, spread_power),
        bounds=bounds,
        method='L-BFGS-B'
        )

    m_opt = res.x
    x_jitter = m_opt * x_max
    return x_jitter

def generate_dot_pos(
        group_data: np.ndarray,
        v_collection:matplotlib.collections.PolyCollection,
        orient:str,
        spread:float,
        spread_strength:float,
        spread_power:float,
        split:bool=False):
    ''' Generating random jittered positions of data points and determine color '''
    verts = copy.copy(v_collection.get_paths()[0].vertices) # Extract the violin shape
    color = copy.copy(v_collection.get_facecolor()[:3]) # Retrieve RGB triplet

    # Treating x and y as y and x respectively for calculating spread, if violins are rotated by 90°
    if orient in ['h', 'y']:
        verts[:, [0, 1]] = verts[:, [1, 0]]
    x_verts, y_verts = verts[:, 0], verts[:, 1]
    
    if not split:
        x_0 = np.mean(x_verts)
        
        # Extracting only left half, since in this case the violin is symmetric
        mask = x_verts <= x_0
        x_verts_masked, y_verts_masked = x_verts[mask], y_verts[mask]
        x_max = np.interp(group_data, y_verts_masked, x_verts_masked-x_0) * spread

        x_jitter = find_optimal_spread_params(group_data, x_max, spread_strength, spread_power, M_BOUNDS[0], M_BOUNDS[1])

    else:
        # Create mask of curved half
        x_0 = collections.Counter(np.round(x_verts, 8)).most_common(1)[0][0] # Most common x-value comes from the "split line"...
        mask = np.round(x_verts, 8) != x_0
        x_verts_masked, y_verts_masked = x_verts[mask], y_verts[mask]

        # Sorting required for interpolation
        sort_idx = np.argsort(y_verts_masked)
        x_sorted, y_sorted,  = x_verts_masked[sort_idx]-x_0, y_verts_masked[sort_idx]
        x_max = np.interp(group_data, y_sorted, x_sorted)

        if (x_verts_masked.sum()<0):
            x_jitter = find_optimal_spread_params(group_data, x_max, spread_strength, spread_power, 0, M_BOUNDS[1])
        else:
            x_jitter = find_optimal_spread_params(group_data, x_max, spread_strength, spread_power, 0, M_BOUNDS[1])

    x_pos = np.full_like(group_data, x_0)

    # Applying data point spread
    if orient in ['v', 'x']:
        x = x_pos+x_jitter
        y = group_data
    elif orient in ['h', 'y']:
        # Reversing swapping x and y for purpose of this calculation, if violins were rotated by 90°
        x = group_data
        y = x_pos + x_jitter
        
    return x, y, color



def fancy_violinplot(
        data: np.ndarray|list[np.ndarray], # TODO enable passing a 2D matrix instead of list (in case of equal group sizes...)
        data_split: np.ndarray|list[np.ndarray]|None = None,
        labels: list[str]|None = None,
        labels_split: list[str] = ['Data 1', 'Data 2'],
        orient: str = 'v',
        palette: str|list[str] = "tab10",
        v_fill: bool = True,
        v_saturation: float = 0.75,
        v_inner: str|None = 'box',
        v_inner_kws: dict = None,
        v_width: float = 0.8,
        v_gap: float = 0.0,
        v_linewidth: float = 1.0,
        v_linecolor: str = 'auto',
        v_cut: float = 0.0,
        v_alpha: float = 0.33,
        v_legend: str | bool = 'auto',
        v_legend_fontsize: float = 12,
        dot_s: float = 30.0,
        dot_marker: matplotlib.markers.MarkerStyle = 'o',
        dot_linewidth: float = 0.0,
        dot_edgecolor: str|None = 'black',
        dot_alpha: float = 1.0,
        spread: float = 0.9,
        spread_strength:float=1.0,
        spread_power:float=4.0,
        ax: matplotlib.axes.Axes|None = None
    ) -> matplotlib.axes.Axes:
    """ Enhanced version of seaborn violinplot, with limited options though...

    This function enhances the seaborn violinplot by additionally plotting each
    data point on top with a certain spread for better visibility.
    Input arguments starting with 'v_' are passed to seaborns' violinplot, while
    input arguments starting with 'dot_' are passed the matplotlib's scatter.
    See their documentation for more information on the parameters!

    Args:
        data (np.ndarray | list[np.ndarray]):  Data as a 1D array, where columns
                            correspond to groups or list of 1D arrays (can be
                            of different sizes)
        data_split (np.ndarray | list[np.ndarray] | None, optional): Data to be
                            compared to 'data' using split violins.
                            Must be of same type and shape as data.
        labels (list[str] | None, optional): Labels for each group.
                            Defaults to None.
        labels_split (list[str], optional): Labels for split violins.
                            Only used, if data_split is provided.
                            Must be a list of length 2.
                            Defaults to ['Data 1', 'Data 2'].
        orient (str, optionsl): Orientation of violins. Use 'v' or 'x' for
                            vertical orientation and 'h' or 'y' for horiuontal.
        palette (str | list[str] | None, optional): Color palette to be used.
                            If this is a string of just one color, each violinplot
                            will be of this color. In case of a seaborn color
                            palette, this palette will be used.
                            Can also be a list of valid matplotlib colors.
                            Defaults to 'tab10'.
        v_fill (bool, optional): If true, a path will be used for violin background.
                            Defaults to True.
        v_saturation (float, optional): Saturation of violin patch.
                            Defaults to 0.75.
        v_inner (str | None, optional): Defines the inner element of violins.
                            Recommended argument: 'box'.
                            If None, no inner element will be displayed.
                            Defaults to 'box'.
        v_inner_kws (dict, optional): Dictionary with optional arguments for
                            the inner element (See seaborn.violinplot doc).
                            Defaults to None.
        v_width (float, optional): Defines width between violins.
                            Defaults to 0.8.
        v_gap (float, optional): Defines gaps between violin halfs.
                            Defaults to 0.0.
        v_linewidth (float, optional): Width of line surrounding violins.
                            Defaults to None.
        v_linecolor (str, optional): Color of line surrounding violins.
                            Defaults to 'auto'.
        v_cut (float, optional): Distance, in units of bandwidth, to extend
                            the density past extreme datapoints. Set to 0 to
                            limit the violin within the data range.
                            Defaults to 0.
        v_alpha (float, optional): Alpha value of the patch.
                            Defaults to 0.33.
        v_legend (str | bool, optional): How to draw the legend by sns.violinplot,
                            if split data is given.
                            Defaults to 'auto'.
        v_legend_fontsize (float, optional): Fontsize of the legend, if split
                            data is given.
                            Defaults to 12.
        dot_s (float, optional): Size of the data dots.
                            Defaults to 30.0.
        dot_marker (matplotlib.markers.MarkerStyle, optional):
                            Matplotlibs' marker style. Defaults to 'o'.
        dot_linewidth (float, optional): Line width of the data dots.
                            Defaults to 0.0.
        dot_edgecolor (str | None, optional): Edge color of the data dots.
                            Defaults to 'black'.
        dot_alpha (float, optional): Alpha value of the data dots.
                            Defaults to 1.0.
        spread (float, optional): Amount of spread of the data dots on the
                            x-axis expressed as a ratio, where 1.0 corresponds
                            to the maximum spread being the bounds of the violin
                            patch. Defaults to 0.9.
        spread_strength (float): The higher the value, the more data points are spread.
                            Defaults to 1.0.
        spread_power (float): Punishes very small distances.
                            Defaults to 4.0.
        ax (matplotlib.axes.Axes| None): Axis to be used for plotting.
                            If None, a new axis will be generated.
                            Defaults to None

    Returns:
        matplotlib.axes.Axes: Axes object
    """
    if isinstance(data, np.ndarray):
        n_groups = data.shape[1] if data.ndim > 1 else 1
    elif isinstance(data, list):
        n_groups = len(data)
    else:
        raise ValueError("Argument 'data' must be np.ndarray or list of np.ndarray")

    split = (data_split is not None) # flag for split data

    v_args_dict = {
        'ax': ax,
        'split': split,
        'orient': orient,
        'palette': palette,
        'saturation': v_saturation,
        'fill': v_fill,
        'inner': v_inner,
        'inner_kws': v_inner_kws,
        'width': v_width,
        'gap': v_gap,
        'linewidth': v_linewidth,
        'linecolor': v_linecolor,
        'cut': v_cut,
        'alpha': v_alpha,
        'legend': v_legend,
        'zorder': 1
    }

    # Violinplot as background
    if not split: # One distribution per violin
        ax = sns.violinplot(data = data, **v_args_dict)
    else: #  Two distributions per violin (split)
        if isinstance(data_split, type(data)):
            if isinstance(data_split, np.ndarray): # One violin only (just one np array provided)
                v_y = np.concatenate((data, data_split), axis=0)
                v_x = labels * data.shape[0] + labels * data_split.shape[0] if labels is not None else None
                hue = [labels_split[0]] * data.shape[0] + [labels_split[1]] * data_split.shape[0]
            elif isinstance(data_split, list): # Multiple violins (list of np arrays provided)
                v_y = np.zeros(0) # One 1D np array required with label vectors for assigning position
                v_x = []
                hue = []
                for ii, (arr, arr_split) in enumerate(zip(data, data_split)):
                    # Appending data for left half (in vertical mode)
                    v_y = np.concatenate((v_y, arr))
                    hue += arr.shape[0] * [labels_split[0]] # Assigning to left side

                    # Appending split data for right half (in vertical mode)
                    v_y = np.concatenate((v_y, arr_split))
                    hue += arr_split.shape[0] * [labels_split[1]] # Assigning to right side

                    # Assigning to same violin for current label
                    if labels is not None:
                        v_x += (arr.shape[0]+arr_split.shape[0]) * [labels[ii]]
                    else:
                        v_x += (arr.shape[0]+arr_split.shape[0]) * [ii]
            else:
                raise ValueError(
                    f"Invalid type for data_split: {type(data_split)}")
        else:
            raise ValueError(
                f"Argument 'data_split' must be of same type as 'data', but "+\
                f"is of type {type(data_split)} instead of {type(data)}")
        
        if orient in ['v', 'x']:
            ax = sns.violinplot(y=v_y, x=v_x, hue=hue, **v_args_dict)
        else:
            ax = sns.violinplot(y=v_x, x=v_y, hue=hue, **v_args_dict)

    # Retrieveing palette colors for plt.scatter
    if isinstance(palette, str):
        if palette in VALID_SNS_PALETTES:
            palette = sns.color_palette(palette, n_groups)
            palette = palette.as_hex() # interpreting as actual palette
        else:
            palette = [palette] * n_groups # interpreting as single color to be repeated

    # Moving boxplot to the front
    for line in ax.get_children():
        if isinstance(line, plt.Line2D):
            line.set_zorder(3)

    # Applying legend fontsize
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(v_legend_fontsize)
    
    violin_collection_list = [c for c in ax.collections if isinstance(c, matplotlib.collections.PolyCollection)]

    dot_args_dict = {
        's': dot_s,
        'marker': dot_marker,
        'alpha': dot_alpha,
        'linewidths': dot_linewidth,
        'edgecolors': dot_edgecolor
    }

    step = 1 if not split else 2
    for ii in range(n_groups):
        group_data = data[:,ii] if isinstance(data, np.ndarray) else data[ii]
        dot_x,dot_y, dot_color = generate_dot_pos(
            group_data, violin_collection_list[ii*step], orient, spread,
            spread_strength, spread_power, split=split)
        ax.scatter(x=dot_x, y=dot_y, color=dot_color, **dot_args_dict)

        if split:
            group_data_split = data_split[:,ii] if isinstance(data_split, np.ndarray) else data_split[ii]
            dot_x_split, dot_y_split, dot_color_split = generate_dot_pos(
                group_data_split, violin_collection_list[ii*step+1], orient, spread, spread_strength, spread_power, split=split)
            ax.scatter(x=dot_x_split, y=dot_y_split, color=dot_color_split, **dot_args_dict)
    
    # Adding group labels to x axis
    if labels is not None:
        match orient:
            case 'v' | 'x':
                ax.set_xticks(range(n_groups))
                ax.set_xticklabels(labels)
            case 'h' | 'y':
                ax.set_yticks(range(n_groups))
                ax.set_yticklabels(labels)
            case _:
                raise ValueError(f"Invalid value for argument 'orient': {orient}")
    
    return ax

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from copt.benchmark import timeit

    # Args for fancy_violinplot
    n_groups = 3
    labels = [f"Group {ii}" for ii in range(n_groups)]
    min_points = 20
    max_points = 40
    max_offset = 4
    min_scale_fac = 0.25
    dot_s = 10

    # Args for fancy_violinplot to be iterated
    spread_strength_list = [0.1, 0.5, 1.0, 2.0]
    spread_power_list = [1.0, 2.0, 3.0, 4.0]

    # Generating example data
    np.random.seed(4) # Making generating data points deeterministic...
    def generate_test_data():
        return np.random.randn(np.random.randint(min_points, max_points)) *\
            max(np.random.rand(1), min_scale_fac) +\
            np.random.rand(1) * max_offset
    data_list = [generate_test_data() for _ in range(n_groups)]
    data_list_split = [generate_test_data() for _ in range(n_groups)]
    
    # Plots for non-split data
    fig, ax = plt.subplots(len(spread_strength_list), len(spread_power_list), figsize=(12, 12))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)
    fig_split, ax_split = plt.subplots(len(spread_strength_list), len(spread_power_list), figsize=(12, 12))
    plt.subplots_adjust(hspace=0.0, wspace=0.0)

    pbar = tqdm(total=len(spread_strength_list)*len(spread_power_list), desc="Generating plots...")

    for ii, spread_strength in enumerate(spread_strength_list):

        ax[ii, 0].set_ylabel('Values')
        ax_split[ii, 0].set_ylabel('Values')

        for jj, spread_power in enumerate(spread_power_list):

            # Non-split data
            _, elapsed_time = timeit(
                fancy_violinplot,
                data=data_list,
                labels=labels,
                orient='v',
                dot_s=dot_s,
                spread_strength=spread_strength,
                spread_power=spread_power,
                ax=ax[ii,jj]
                )
            ax[ii, jj].text(
                0.02, 0.98,
                f'spread_strength={spread_strength}\nspread_power={spread_power}\nelapsed time={elapsed_time:.2f} s',
                transform=ax[ii, jj].transAxes,
                ha='left', va='top',
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5)
                )
            
            # Split data
            _, elapsed_time_split = timeit(
                fancy_violinplot,
                data=data_list,
                data_split=data_list_split,
                labels=labels,
                labels_split=['l', 'r'],
                orient='v',
                dot_s=dot_s,
                spread_strength=spread_strength,
                spread_power=spread_power,
                ax=ax_split[ii,jj]
                )
        
            ax_split[ii, jj].text(
                0.02, 0.98,
                f'spread_strength={spread_strength}\nspread_power={spread_power}\nelapsed time={elapsed_time_split:.2f} s',
                transform=ax_split[ii, jj].transAxes,
                ha='left', va='top',
                fontsize=9,
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5)
                )
            
            pbar.update()
            
        ax[ii,jj].set_xlabel('Categories')
        ax_split[ii,jj].set_xlabel('Categories')

    fig.suptitle('Regular fancy violin plots with different spreads with random example data')
    fig_split.suptitle('Fancy violin plots with different spreads with random example split data')

    plt.show()