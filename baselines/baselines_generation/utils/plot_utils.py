"""Plot utility functions.

selected colors with https://www.jianshu.com/p/c6660e75f773
fill(files, 'salmon', 'red', 'unify')
fill(files1, 'lightgreen', 'green', 'upo3')
fill(files2, 'cyan', 'blue', 'ppo')
fill(files3, 'orchid', 'purple', 'acktr')
fill(files4, 'grey', 'black', 'a2c')
"""

import matplotlib.pyplot as plt

colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
        '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

# def boxplot(x_data, y_data, base_color="#539caf", median_color="#297083", x_label="", y_label="", title=""):
def boxplot(x_data, y_data, base_color="cornflowerblue", median_color="darkorange", x_label="", y_label="", title=""):
    _, ax = plt.subplots()

    # Draw boxplots, specifying desired style
    ax.boxplot(y_data
                # patch_artist must be True to control box fill
                , patch_artist = True
                # Properties of median line
                , medianprops = {'color': median_color}
                # Properties of box
                , boxprops = {'color': base_color, 'facecolor': base_color}
                # Properties of whiskers
                , whiskerprops = {'color': base_color}
                # Properties of whisker caps
                , capprops = {'color': base_color})

    # By default, the tick label starts at 1 and increments by 1 for
    # each box drawn. This sets the labels to the ones we want
    ax.set_xticklabels(x_data)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    # ax.set(ylim=(0, 100))
    plt.show()

def temporal_lineplot(x_data, y_data, x_label="", y_label="", title="", lim=None):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    t = list(range(24))
    line_num = 0

    for user in x_data:
        ax.plot(t, y_data[line_num], lw = 2, color = colors[line_num%10], alpha = 0.8, label=user)
        # ax.text(23, y_data[line_num][-1], user, horizontalalignment='left', size='small', color=colors[line_num])
        line_num += 1

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc=0, ncol=2)
    if lim is not None:
        ax.set(ylim=(0, lim))
    plt.show()

def distribution_lineplot(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    fig_row = 4
    fig_col = 6
    _, ax = plt.subplots(fig_row, fig_col)

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line

    for t in range(24):
        fig_i = int(t/fig_col)
        fig_j = int(t%fig_col)
        print('try to plot %d %d', fig_i, fig_j)
        ax[fig_i][fig_j].plot(x_data, y_data[t], lw = 2, alpha = 0.8, label=t)
        ax[fig_i][fig_j].set_title("")
        ax[fig_i][fig_j].set_xlabel(x_label)
        ax[fig_i][fig_j].set_ylabel(y_label)
        ax[fig_i][fig_j].legend(loc=0, ncol=2)
        ax[fig_i][fig_j].set(ylim=(0, 0.1))

    # Label the axes and provide a title
    plt.show()

def distribution_lineplot_in_one(x_data, y_data, x_label="", y_label="", title=""):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line

    for t in range(24):
        ax.plot(x_data, y_data[t], lw = 2, alpha = 0.8, label=t)
      
    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc=0, ncol=2)
    plt.show()

# def plot_source_distribution(value_list, fig_name='data_dist', range_=None, bins_=100,
#         title_='Frequency of records of users' , x_label='users', y_label='number of records (rows)'):
#     plt.style.use('bmh')
#     print(len(value_list))
    
#     fig = plt.figure(figsize=(11,3))
#     _ = plt.title(title_)
#     _ = plt.xlabel(x_label)
#     _ = plt.ylabel(y_label)
#     _ = plt.hist(value_list, histtype='stepfilled')

#     # fig.savefig(fig_name)
#     plt.show()

def plot_source_distribution(y_data, x_label="users", y_label="log number of records (rows)", title="number of records of users"):
    # Create the plot object
    _, ax = plt.subplots()

    # Plot the best fit line, set the linewidth (lw), color and
    # transparency (alpha) of the line
    x_data = list(range(len(y_data)))
    ax.plot(x_data, y_data, lw = 2, color = colors[0], alpha = 0.8)

    # Label the axes and provide a title
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.show()
    # plt.savefig('data_draw.png')