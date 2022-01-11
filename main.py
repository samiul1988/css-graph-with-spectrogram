### Author: Samiul Choudhury
### Date: 2022-01-07

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import math

START_FREQ_LABEL = 'LZeq 6.3Hz'
END_FREQ_LABEL = 'LZeq 20kHz'
TIME_LABEL = 'Start Time'
BB_LABEL = 'LAeq'
FREQ_AXIS_LABELS = ['16Hz', '31.5Hz', '63Hz', '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz','8kHz']
BB_YAXIS_LOWER_LIMIT = 20
BB_YAXIS_UPPER_LIMIT = 100
START_TIME_DAY = '07:00:00'
END_TIME_DAY = '21:59:00'
START_TIME_NIGHT = '22:00:00'
END_TIME_NIGHT = '06:59:00'
OUTPUT_FILE_NAME = "output.pdf"

file_dir_logged_bb = r'H:\PythonProjects\CssGraphTemplate\2250-5 003_LoggedBB.txt'
file_dir_logged_spectra = r'H:\PythonProjects\CssGraphTemplate\2250-5 003_LoggedSpectra.txt'
# file_dir_marker = r'H:\PythonProjects\CssGraphTemplate\2250-5 003_Markers.txt'

# file_dir_logged_bb = r'H:\PythonProjects\CssGraphTemplate\Project002_LoggedBB.txt'
# file_dir_logged_spectra = r'H:\PythonProjects\CssGraphTemplate\Project002_LoggedSpectra.txt'


def generate_plot(
        index,
        day_night_state,
        graph_data_bb,
        graph_data_spectral,
        graph_data_residual,
        graph_data_marker,
        labels_freq,
        labels_time,
        labels_marker,
        chart_title,
        app_num,
        date_title
):
    ## Plot Data
    fig = plt.figure(index)
    fig.suptitle('Comprehensive Sound Survey - {} \n '.format(chart_title), y=0.94, fontweight='bold', fontsize=14)
    fig.text(0.5, 0.89, "Figure {}{}        {} {}        Date: {}".format(app_num, index, day_night_state, int((index + 1) / 2), date_title),
             fontsize=11, horizontalalignment='center')
    fig.set_size_inches(9,9)
    fig.set_dpi(100)
    gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[2,1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    ## Broadband Data Axis
    ax1.plot(labels_time, graph_data_bb, color=(224/255, 223/255, 220/255), linewidth=1)
    ax1.plot(labels_time, graph_data_residual, color='black', linewidth=2.5, label="Residual Leq (dBA)")
    for i, marker in enumerate(labels_marker):
        ax1.fill_between(labels_time, graph_data_marker.loc[:, marker], BB_YAXIS_LOWER_LIMIT, color='C{}'.format(i), alpha=0.3, label=marker)
    # ax1.fill_between(labels_time, graph_data_bb, BB_YAXIS_LOWER_LIMIT, color='C0', alpha=0.3)
    ax1.grid(which='both', axis='y', linewidth=0.6)
    ax1.set_ylabel('Sound Levels (dBA)', fontdict={'fontsize':9, 'fontweight': 'bold'})
    ax1.get_xaxis().set_visible(False)
    ax1.set_yticks(np.arange(BB_YAXIS_LOWER_LIMIT, BB_YAXIS_UPPER_LIMIT, step=5, dtype=int), minor=True)
    ax1.legend(ncol=3, fancybox=True, shadow=True)
    ## Spectral Data Axis
    ax2.pcolormesh(labels_time, labels_freq, graph_data_spectral)
    ax2.yaxis.set_ticks(FREQ_AXIS_LABELS)
    ax2.set_xlabel('Time', fontdict={'fontsize':9, 'fontweight': 'bold'})
    ax2.set_ylabel('Frequency', fontdict={'fontsize':9, 'fontweight': 'bold'})
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(int(start), int(end), 30))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: pd.to_datetime(labels_time[int(x)]).strftime("%H:%M")))
    for label in ax2.get_xticklabels(which='major'):
        label.set(rotation=90)
    return fig
    # plt.show()
    # ax1.remove()
    # ax2.remove()


try:
    db_data_logged_bb = pd.read_csv(file_dir_logged_bb, delimiter='\t')
    db_data_logged_spectra = pd.read_csv(file_dir_logged_spectra, delimiter='\t')
    # db_data_marker = pd.read_csv(file_dir_marker, delimiter='\t')

    df_logged_bb = pd.DataFrame(db_data_logged_bb)
    df_logged_spectra = pd.DataFrame(db_data_logged_spectra)
    # df_marker = pd.DataFrame(db_data_marker)
except:
    print("Could not read data from files")
    exit()

# Extract Broadband Data and Markers
bb_data = df_logged_bb.loc[:, BB_LABEL]
sound_marker_index = df_logged_bb.columns.get_loc('Sound')
marker_data = df_logged_bb.iloc[:, sound_marker_index - 5 : sound_marker_index]
marker_labels = np.array(marker_data.columns)
# Compute Residual Data
total_marker_data = marker_data.sum(axis=1)
total_marker_data = total_marker_data.map(lambda t: np.nan if t > 0 else 1.0)
residual_data = bb_data.mul(total_marker_data, axis=0)
# Process Marker Data for Graphs
marker_data = marker_data.mul(bb_data, axis=0)
marker_data = marker_data.replace(0, np.nan)

# print(total_marker_data)
# print(type(total_marker_data))
# print(residual_data)
# marker2_data = df_logged_bb.iloc[:, sound_marker_index - 5]
# marker3_data = df_logged_bb.iloc[:, sound_marker_index - 5]
# marker4_data = df_logged_bb.iloc[:, sound_marker_index - 5]
# marker5_data = df_logged_bb.iloc[:, sound_marker_index - 5]

# print(marker1_data)
# bb_data2 = bb_data.copy()
# bb_data[100:400] = np.nan



# Extract Spectral Data
col_idx_start_freq = df_logged_spectra.columns.get_loc(START_FREQ_LABEL)
col_idx_end_freq = df_logged_spectra.columns.get_loc(END_FREQ_LABEL) + 1
spectral_data = df_logged_spectra.iloc[:, col_idx_start_freq: col_idx_end_freq].transpose()

# Extract Time information
time_data = df_logged_bb.loc[:, TIME_LABEL]
time_data_dt = pd.to_datetime(time_data).dt.floor('Min')
unique_dates = time_data_dt.map(lambda t: t.date()).unique()

# print(time_data[0])
# print(time_data_dt_str[0])

# Prepend data if needed
data_start_time = time_data_dt[0]
data_end_time = time_data_dt[time_data_dt.size - 1]
start_time_7AM = pd.to_datetime("{} {}".format(data_start_time.date(), START_TIME_DAY))
start_time_10PM = pd.to_datetime("{} {}".format(data_start_time.date(), START_TIME_NIGHT))
end_time_7AM = pd.to_datetime("{} {}".format(data_end_time.date(), START_TIME_DAY))
end_time_10PM = pd.to_datetime("{} {}".format(data_end_time.date(), START_TIME_NIGHT))

minutes_diff_from_7AM = int((data_start_time - start_time_7AM).total_seconds() / 60.0)
if minutes_diff_from_7AM < 0:
    time_data_to_prepend = pd.Series(pd.date_range("{} {}".format(data_start_time.date() - pd.DateOffset(1), START_TIME_NIGHT), periods=540+minutes_diff_from_7AM, freq="T"))
else:
    minutes_diff_from_10PM = int((data_start_time - start_time_10PM).total_seconds() / 60.0)
    if minutes_diff_from_10PM >= 0:
        time_data_to_prepend = pd.Series(pd.date_range(start_time_10PM, periods=minutes_diff_from_10PM, freq="T"))
    else:
        time_data_to_prepend = pd.Series(pd.date_range(start_time_7AM, periods=minutes_diff_from_7AM, freq="T"))

# print(time_data_to_prepend)
time_data_dt = time_data_to_prepend.append(time_data_dt, ignore_index=True)
bb_data = pd.Series(np.nan, index=range(time_data_to_prepend.size)).append(bb_data, ignore_index=True)
spectral_data_to_prepend = pd.DataFrame(np.nan, index=spectral_data.index, columns=range(time_data_to_prepend.size))
spectral_data = pd.concat([spectral_data_to_prepend, spectral_data], axis=1).reindex(spectral_data.index)
residual_data = pd.Series(np.nan, index=range(time_data_to_prepend.size)).append(residual_data, ignore_index=True)
marker_data_to_prepend = pd.DataFrame(np.nan, index=range(time_data_to_prepend.size), columns=marker_data.columns)
marker_data = pd.concat([marker_data_to_prepend, marker_data], axis=0, ignore_index=True)

# Append data if needed
minutes_diff_from_7AM = int((data_end_time - end_time_7AM).total_seconds() / 60.0)
if minutes_diff_from_7AM < 0:
    time_data_to_append = pd.Series(pd.date_range(data_end_time + pd.Timedelta(minutes=1), periods=(-1* minutes_diff_from_7AM) - 1, freq="T"))
else:
    minutes_diff_from_10PM = int((data_end_time - end_time_10PM).total_seconds() / 60.0)
    if minutes_diff_from_10PM < 0:
        time_data_to_append = pd.Series(pd.date_range(data_end_time + pd.Timedelta(minutes=1), periods=(-1 * minutes_diff_from_10PM) - 1, freq="T"))
    else:
        time_data_to_append = pd.Series(pd.date_range(data_end_time + pd.Timedelta(minutes=1), periods=540 - minutes_diff_from_10PM - 1 , freq="T"))

# print(time_data_to_append)
time_data_dt = time_data_dt.append(time_data_to_append, ignore_index=True)
bb_data = bb_data.append(pd.Series(np.nan, index=range(time_data_to_append.size)), ignore_index=True)
spectral_data_to_append = pd.DataFrame(np.nan, index=spectral_data.index, columns=range(time_data_to_append.size))
spectral_data = pd.concat([spectral_data, spectral_data_to_append], axis=1).reindex(spectral_data.index)
residual_data = residual_data.append(pd.Series(np.nan, index=range(time_data_to_append.size)), ignore_index=True)
marker_data_to_append = pd.DataFrame(np.nan, index=range(time_data_to_append.size), columns=marker_data.columns)
marker_data = pd.concat([marker_data, marker_data_to_append], axis=0, ignore_index=True)

# print(time_data_dt)
# print(bb_data)
# print(type(spectral_data), spectral_data.shape)

# Get and format frequency labels
freq_labels = np.array(df_logged_spectra.columns[col_idx_start_freq: col_idx_end_freq])
formatter = np.vectorize(lambda x: x.replace("LZeq ", ""))
freq_labels = formatter(freq_labels)

time_data_dt_str = time_data_dt.map(lambda t: str(t))
time_labels = np.array(time_data_dt_str)

# Run loop for day and nighttime plots
idx = 0
counter = 1
state = "Day" if str(time_data_dt[0].time()) == START_TIME_DAY else "Night"
date_text = ""

with PdfPages(OUTPUT_FILE_NAME) as pdf:
    while idx < time_data_dt.size:
        # if idx == 0:
        if state == "Day":
            labels_time = time_labels[idx: (idx + 900)]
            graph_data_bb = bb_data[idx: (idx + 900)]
            graph_data_spectral = spectral_data.iloc[:, idx: (idx + 900)]
            graph_data_residual = residual_data[idx: (idx + 900)]
            graph_data_marker = marker_data.iloc[idx: (idx + 900), :]
            date_text = time_data_dt[idx].strftime("%b %d, %Y")
            idx = idx + 900
        else:
            labels_time = time_labels[idx: (idx + 540)]
            graph_data_bb = bb_data[idx: (idx + 540)]
            graph_data_spectral = spectral_data.iloc[:, idx: (idx + 540)]
            graph_data_residual = residual_data[idx: (idx + 540)]
            graph_data_marker = marker_data.iloc[idx: (idx + 540), :]
            date_text = "{} - {}".format(time_data_dt[idx].strftime("%b %d"), (time_data_dt[idx] + pd.Timedelta(days=1)).strftime("%b %d, %Y"))
            idx = idx + 540
        measured_leq = math.log10(graph_data_bb.map(lambda t: pow(10, t/10)).mean())*10
        # fig = generate_plot(counter, state, graph_data_bb, graph_data_spectral, graph_data_residual, graph_data_marker, freq_labels, labels_time, marker_labels, "Residence 1", "E", date_text)
        # pdf.savefig(fig)
        # plt.close()
        state = "Night" if state == "Day" else "Day"
        counter += 1


# ## Plot Data
# fig = plt.figure()
# fig.suptitle('Comprehensive Sound Survey (dBA) \n Day 1', y=0.94, fontsize='x-large', fontweight='bold')
# fig.set_size_inches(9,9)
# fig.set_dpi(100)
# gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[2,1])
# ax1 = fig.add_subplot(gs[0])
# ax2 = fig.add_subplot(gs[1], sharex=ax1)
#
# ## Broadband Data Axis
# # ax1.plot(bb_data2, color=(224/255, 223/255, 220/255), linewidth=1)
# ax1.plot(bb_data, color='black', linewidth=2.5)
# ax1.fill_between(np.arange(time_labels.size), bb_data, BB_YAXIS_LOWER_LIMIT, color='C0', alpha=0.3)
# ax1.grid(which='both', axis='y', linewidth=0.6)
# ax1.set_ylabel('Sound Levels (dBA)', fontdict={'fontsize':'large', 'fontweight': 'bold'})
# ax1.get_xaxis().set_visible(False)
# ax1.set_yticks(np.arange(BB_YAXIS_LOWER_LIMIT, BB_YAXIS_UPPER_LIMIT, step=5, dtype=int), minor=True)
#
# ## Spectral Data Axis
# ax2.pcolormesh(time_labels, freq_labels, spectral_data)
# ax2.yaxis.set_ticks(FREQ_AXIS_LABELS)
# ax2.set_xlabel('Time', fontdict={'fontsize':'large', 'fontweight': 'bold'})
# ax2.set_ylabel('Frequency', fontdict={'fontsize':'large', 'fontweight': 'bold'})
# start, end = ax2.get_xlim()
# ax2.xaxis.set_ticks(np.arange(int(start), int(end), 30))
# ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: pd.to_datetime(time_labels[int(x)]).strftime("%H:%M")))
# for label in ax2.get_xticklabels(which='major'):
#     label.set(rotation=90)

# plt.show()

# fig.savefig('test.pdf')


# # Hide x labels and tick labels for all but bottom plot.
# for ax in axs:
#     ax.label_outer()

# def format_time(x, pos=None):
#     return pd.to_datetime(time_labels[x]).strftime("%H:%M")



# print(np.array(df.columns))
# print(df.iloc[:, [0,1]])
# print(df.columns.get_loc('Project Name'))

