import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from qiskit.visualization import plot_histogram


def prepare_plot(x, y, title, xlabel, ylabel, lineformat, label=None, pointformat=None, xticks=None, yticks=None,
                 log_scale=None, size=(5, 5), ax=None):
    # If ax is given then plot in the same, skipping all settings
    if ax is not None:
        ax.plot(x, y, lineformat, label=label)
        if pointformat is not None:
            ax.plot(x, y, pointformat)
        return ax
    fig, ax = plt.subplots(1, 1, figsize=size)
    ax.plot(x, y, lineformat, label=label)
    if pointformat is not None:
        ax.plot(x, y, pointformat)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if log_scale:
        ax.set_yscale('log')
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    return ax


def prepare_plot_iterations(x, y, reasons, errors, title, xlabel, ylabel, xticks=None, yticks=None, size=(15, 15)):
    error_max = max(errors)
    err_array = errors.copy()

    # Make error better distributed in the colormap
    ticks_to_add = []
    for i in range(4, 0, -1):
        unit = 10 ** (-i)
        ticks_to_add.extend([round(j * unit, i + 1) for j in range(1, 10, 2)])
    err_array.extend(ticks_to_add)
    err_array = sorted(err_array)

    cmap = plt.get_cmap('YlGnBu')
    sampling = np.linspace(0, min(error_max, 1), len(err_array))
    colors = cmap(sampling)

    fig, ax = plt.subplots(1, 1, figsize=size)
    rects = ax.barh(x, y)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)

    for i, rect in enumerate(rects):
        width = int(rect.get_width())
        if width < 1250:
            # Shift the text to the right side of the right edge
            xloc = 5
            # Black against white background
            clr = 'black'
            align = 'left'
        else:
            # Shift the text to the left side of the right edge
            xloc = -5
            # White on magenta
            clr = 'white'
            align = 'right'
        yloc = rect.get_y() + rect.get_height() / 2

        text = f'It: {y[i]} - {reasons[i]}'
        # Identify the color in the colormap
        idx = err_array.index(errors[i])
        rect.set_color(colors[idx])

        ax.annotate(text, xy=(width, yloc), xytext=(xloc, 0),
                    textcoords="offset points",
                    horizontalalignment=align, verticalalignment='center',
                    color=clr, weight='bold', clip_on=True)

    # Create ticks for color bar excluding artificially added values
    labels_err = err_array.copy()
    for el in ticks_to_add:
        labels_err[labels_err.index(el)] = None

    cb = fig.colorbar(cm.ScalarMappable(cmap=cmap), ticks=sampling)
    cb.ax.set_yticklabels(labels_err)
    cb.ax.set_ylabel("Errors", fontsize=15)

    return ax


def prepare_state_vector_plots(state_l, target_state, best_state, layer, best_layer, size):
    fig, ax = plt.subplots(1, 3, figsize=size)
    plot_histogram(state_l.probabilities_dict(), sort='asc', ax=ax[0])
    ax[0].set_title(f"State for {layer} layer(s)", fontsize=20)
    plot_histogram(target_state.probabilities_dict(), sort='asc', ax=ax[1])
    ax[1].set_title(f"Target State", fontsize=20)
    plot_histogram(best_state.probabilities_dict(), sort='asc', ax=ax[2])
    ax[2].set_title(f"Best State - {best_layer} layer(s)", fontsize=20)
