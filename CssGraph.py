"""
Author: Samiul Choudhury
Copyright: Samiul Choudhury
Last Updated: 2022-01-11
"""

""" 
Python Script for Generating CSS Graphs

"""

# # Import Modules
import os
import time
import xlwings as xw
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
# FREQ_AXIS_LABELS = ['16Hz', '31.5Hz', '63Hz', '125Hz', '250Hz', '500Hz', '1kHz', '2kHz', '4kHz','8kHz']
FREQ_AXIS_LABELS = ['16', '31.5', '63', '125', '250', '500', '1k', '2k', '4k','8k']
BB_YAXIS_LOWER_LIMIT = 20
BB_YAXIS_UPPER_LIMIT = 100
START_TIME_DAY = '07:00:00'
END_TIME_DAY = '21:59:00'
START_TIME_NIGHT = '22:00:00'
END_TIME_NIGHT = '06:59:00'
OUTPUT_FILE_NAME = "CSS_Graph_Output"

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
        date_title,
        year_title
):
    ## Plot Data
    fig = plt.figure(index)
    fig.suptitle('{} \n '.format(chart_title), y=0.94, fontweight='bold', fontsize=14)
    fig.text(0.5, 0.89, "Figure {}{}        {} {}        Date: {}, {}".format(app_num, index, day_night_state, int((index + 1) / 2), date_title, year_title),
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
    ax2.set_ylabel('Frequency (Hz)', fontdict={'fontsize':9, 'fontweight': 'bold'})
    start, end = ax2.get_xlim()
    ax2.xaxis.set_ticks(np.arange(int(start), int(end), 30))
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,pos: pd.to_datetime(labels_time[int(x)]).strftime("%H:%M")))
    for label in ax2.get_xticklabels(which='major'):
        label.set(rotation=90)
    return fig


def main():
    wb = xw.Book.caller()
    ws = wb.sheets["Info"]
    displayMsgHandler = wb.macro('DisplayMessage')
    selectFileHandler = wb.macro('SelectFile')
    getCurrentDirHandler = wb.macro('GetCurrentDir')
    # displayInputFormHandler = wb.macro('DisplayForm')
    # getColIndexFunctionHandler = wb.macro('GetColIndex')
    # getRowIndexFunctionHandler = wb.macro('GetRowIndex')

    # Select Broadband and Spectral Data Files
    file_dir_logged_bb = selectFileHandler("Select a Broadband Data (.txt) File")
    file_dir_logged_spectra = selectFileHandler("Select a Spectral Data (.txt) File")
    if file_dir_logged_bb == "" or file_dir_logged_spectra == "":
        displayMsgHandler("Broadband/Spectral Data File Path(s) are invalid")
        return

    # Get relevant metadata
    chart_title = ws.range("B1").value
    appendix_num = ws.range("B2").value

    if chart_title == "" or appendix_num == "":
        displayMsgHandler("Chart title / Appendix Number is invalid")
        return

    try:
        db_data_logged_bb = pd.read_csv(file_dir_logged_bb, delimiter='\t')
        db_data_logged_spectra = pd.read_csv(file_dir_logged_spectra, delimiter='\t')

        df_logged_bb = pd.DataFrame(db_data_logged_bb)
        df_logged_spectra = pd.DataFrame(db_data_logged_spectra)
    except:
        displayMsgHandler("Could not read data from files")
        return

    ws.range("A4").value = "Process running..."
    # Extract Broadband Data and Markers
    bb_data = df_logged_bb.loc[:, BB_LABEL]
    sound_marker_index = df_logged_bb.columns.get_loc('Sound')
    marker_data = df_logged_bb.iloc[:, sound_marker_index - 5: sound_marker_index]
    marker_labels = np.array(marker_data.columns)

    # Compute Residual Data
    total_marker_data = marker_data.sum(axis=1)
    total_marker_data = total_marker_data.map(lambda t: np.nan if t > 0 else 1.0)
    residual_data = bb_data.mul(total_marker_data, axis=0)

    # Process Marker Data for Graphs
    marker_data = marker_data.mul(bb_data, axis=0)
    marker_data = marker_data.replace(0, np.nan)

    # Extract Spectral Data
    col_idx_start_freq = df_logged_spectra.columns.get_loc(START_FREQ_LABEL)
    col_idx_end_freq = df_logged_spectra.columns.get_loc(END_FREQ_LABEL) + 1
    spectral_data = df_logged_spectra.iloc[:, col_idx_start_freq: col_idx_end_freq].transpose()

    # Extract Time information
    time_data = df_logged_bb.loc[:, TIME_LABEL]
    time_data_dt = pd.to_datetime(time_data).dt.floor('Min')

    # Prepend data if needed
    data_start_time = time_data_dt[0]
    data_end_time = time_data_dt[time_data_dt.size - 1]
    start_time_7AM = pd.to_datetime("{} {}".format(data_start_time.date(), START_TIME_DAY))
    start_time_10PM = pd.to_datetime("{} {}".format(data_start_time.date(), START_TIME_NIGHT))
    end_time_7AM = pd.to_datetime("{} {}".format(data_end_time.date(), START_TIME_DAY))
    end_time_10PM = pd.to_datetime("{} {}".format(data_end_time.date(), START_TIME_NIGHT))

    minutes_diff_from_7AM = int((data_start_time - start_time_7AM).total_seconds() / 60.0)
    if minutes_diff_from_7AM < 0:
        time_data_to_prepend = pd.Series(
            pd.date_range("{} {}".format(data_start_time.date() - pd.DateOffset(1), START_TIME_NIGHT),
                          periods=540 + minutes_diff_from_7AM, freq="T"))
    else:
        minutes_diff_from_10PM = int((data_start_time - start_time_10PM).total_seconds() / 60.0)
        if minutes_diff_from_10PM >= 0:
            time_data_to_prepend = pd.Series(pd.date_range(start_time_10PM, periods=minutes_diff_from_10PM, freq="T"))
        else:
            time_data_to_prepend = pd.Series(pd.date_range(start_time_7AM, periods=minutes_diff_from_7AM, freq="T"))

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
        time_data_to_append = pd.Series(
            pd.date_range(data_end_time + pd.Timedelta(minutes=1), periods=(-1 * minutes_diff_from_7AM) - 1, freq="T"))
    else:
        minutes_diff_from_10PM = int((data_end_time - end_time_10PM).total_seconds() / 60.0)
        if minutes_diff_from_10PM < 0:
            time_data_to_append = pd.Series(
                pd.date_range(data_end_time + pd.Timedelta(minutes=1), periods=(-1 * minutes_diff_from_10PM) - 1,
                              freq="T"))
        else:
            time_data_to_append = pd.Series(
                pd.date_range(data_end_time + pd.Timedelta(minutes=1), periods=540 - minutes_diff_from_10PM - 1,
                              freq="T"))

    time_data_dt = time_data_dt.append(time_data_to_append, ignore_index=True)
    bb_data = bb_data.append(pd.Series(np.nan, index=range(time_data_to_append.size)), ignore_index=True)
    spectral_data_to_append = pd.DataFrame(np.nan, index=spectral_data.index, columns=range(time_data_to_append.size))
    spectral_data = pd.concat([spectral_data, spectral_data_to_append], axis=1).reindex(spectral_data.index)
    residual_data = residual_data.append(pd.Series(np.nan, index=range(time_data_to_append.size)), ignore_index=True)
    marker_data_to_append = pd.DataFrame(np.nan, index=range(time_data_to_append.size), columns=marker_data.columns)
    marker_data = pd.concat([marker_data, marker_data_to_append], axis=0, ignore_index=True)

    # Get and format frequency labels
    freq_labels = np.array(df_logged_spectra.columns[col_idx_start_freq: col_idx_end_freq])
    formatter = np.vectorize(lambda x: x.replace("LZeq ", "").replace("Hz", ""))
    freq_labels = formatter(freq_labels)

    time_data_dt_str = time_data_dt.map(lambda t: str(t))
    time_labels = np.array(time_data_dt_str)

    # Run loop for day and nighttime plots
    idx = 0
    counter = 1
    state = "Day" if str(time_data_dt[0].time()) == START_TIME_DAY else "Night"
    date_text = ""
    output_pdf_file_path = "{}\{}.pdf".format(getCurrentDirHandler(), OUTPUT_FILE_NAME)

    with PdfPages(output_pdf_file_path) as pdf:
        while idx < time_data_dt.size:
            if state == "Day":
                labels_time = time_labels[idx: (idx + 900)]
                graph_data_bb = bb_data[idx: (idx + 900)]
                graph_data_spectral = spectral_data.iloc[:, idx: (idx + 900)]
                graph_data_residual = residual_data[idx: (idx + 900)]
                graph_data_marker = marker_data.iloc[idx: (idx + 900), :]
                date_text = time_data_dt[idx].strftime("%b %d")
                year_text = time_data_dt[idx].strftime("%Y")
                idx = idx + 900
            else:
                labels_time = time_labels[idx: (idx + 540)]
                graph_data_bb = bb_data[idx: (idx + 540)]
                graph_data_spectral = spectral_data.iloc[:, idx: (idx + 540)]
                graph_data_residual = residual_data[idx: (idx + 540)]
                graph_data_marker = marker_data.iloc[idx: (idx + 540), :]
                date_text = "{} - {}".format(time_data_dt[idx].strftime("%b %d"),
                                             (time_data_dt[idx] + pd.Timedelta(days=1)).strftime("%b %d"))
                year_text = time_data_dt[idx].strftime("%Y")
                idx = idx + 540
            fig = generate_plot(counter, state, graph_data_bb, graph_data_spectral, graph_data_residual,
                                graph_data_marker, freq_labels, labels_time, marker_labels, chart_title, appendix_num,
                                date_text, year_text)
            pdf.savefig(fig)
            plt.close()

            # Update CSS Results Table
            bb_leq_avg = math.log10(graph_data_bb.map(lambda t: pow(10, t / 10)).mean()) * 10
            residual_leq_avg = math.log10(graph_data_residual.map(lambda t: pow(10, t / 10)).mean()) * 10
            psl = 50.0
            wb.sheets["Table CSS"].range("A{}".format(counter + 2)).value = "'{} {}".format(state, int((counter + 1) / 2))
            wb.sheets["Table CSS"].range("B{}".format(counter + 2)).value = "'{}".format(date_text)
            wb.sheets["Table CSS"].range("C{}".format(counter + 2)).value = bb_leq_avg
            wb.sheets["Table CSS"].range("D{}".format(counter + 2)).value = (graph_data_bb.size - graph_data_bb.isna().sum()) / 60
            wb.sheets["Table CSS"].range("E{}".format(counter + 2)).value = residual_leq_avg
            wb.sheets["Table CSS"].range("F{}".format(counter + 2)).value = (graph_data_residual.size - graph_data_residual.isna().sum()) / 60
            wb.sheets["Table CSS"].range("G{}".format(counter + 2)).value = psl if state == "Day" else psl - 10
            wb.sheets["Table CSS"].range("H{}".format(counter + 2)).value = "No" if residual_leq_avg > psl else "Yes"
            wb.sheets["Table CSS"].range("B2").value = "Date \n ({})".format(year_text)

            state = "Night" if state == "Day" else "Day"
            counter += 1

    ws.range("A4").value = "Process finished!"
    time.sleep(2)
    ws.range("A4").value = ""
