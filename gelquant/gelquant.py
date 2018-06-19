from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os
import shutil
from natsort import natsorted, ns
from scipy.integrate import trapz
from scipy.stats import norm
from scipy.optimize import curve_fit


def image_cropping(path, x1, y1, x2, y2):

    """
    Crop image in preparation for gel analysis.
    x and y values correspond to two points in gel
    at which cropping will happen - (x1,y1) is the top
    left point, while (x2,y2) is the bottom right point,
    thus cropping the image between the two points.
    """

    image = Image.open(path)
    plt.figure(figsize=(7,14))
    plt.subplot(121)
    plt.imshow(image)
    plt.title("original image")

    image2 = image.crop((x1, y1, x2, y2))
    plt.subplot(122)
    plt.imshow(image2)
    plt.title("cropped image")

    plt.tight_layout()
    plt.show()

    return image2

def lane_parser(img, lanes, groups, baseline1, baseline2, tolerance=0.1, plot_output=False):

    if "tmp" not in os.listdir():
        os.mkdir("tmp")

    image_array = np.array(img)

    lane_list = np.arange(lanes)

    for i in range(len(lane_list)):
        image_slice = img.crop((len(image_array[0])*i/lanes, 0,
                         len(image_array[0])*(i+1)/lanes, len(image_array)))
        image_slice.save("tmp/lane-" + str(lane_list[i]+1) + ".png","PNG")

    image_list = []

    for i in sorted(os.listdir("tmp")):
        if ".png" in i:
            image_list.append(i)

    image_list = natsorted(image_list, key=lambda y: y.lower())

    final_data = []

    for i in range(len(image_list)):

        data = np.array(Image.open("tmp/" + image_list[i]))

        all_intensities = []

        for j in range(len(data)):
            row_intensities = []
            for k in range(len(data[0])):
                pixel = data[j,k]
                intensity = 1-(0.2126*pixel[0]/255 + 0.7152*pixel[1]/255 + 0.0722*pixel[2]/255)
                row_intensities.append(intensity)
            all_intensities.append(row_intensities)

        final_intensities = []

        for i in range(len(all_intensities)):
            x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), len(all_intensities[i]))
            weights = norm.pdf(x)
            ave_intensity = np.average(all_intensities[i], weights=weights*sum(weights))
            final_intensities.append(ave_intensity)

        final_intensities = np.array(final_intensities) - np.mean(final_intensities[baseline1:baseline2])

        final_data.append(final_intensities)

    peakzero_xs = []
    peakzero_ys = []

    for i in range(groups):
        initial_peak = max(final_data[int(i*len(final_data)/groups)])
        peakzero_ys.append(initial_peak)
        for j in range(len(final_data[int(i*len(final_data)/groups)])):
            if initial_peak == final_data[int(i*len(final_data)/groups)][j]:
                peakzero_xs.append(j)

    all_bounds = []

    for i in range(groups):

        peak = peakzero_ys[i]
        bounds = []

        for j in range(len(final_data[0])):
            if final_data[int(i*len(final_data)/groups)][j] < tolerance*peak:
                continue
            if final_data[int(i*len(final_data)/groups)][j] > tolerance*peak:
                bounds.append(j)

        lower_bound = bounds[0]
        upper_bound = bounds[-1]

        for k in range(int(len(final_data)/groups)):
            all_bounds.append([lower_bound, upper_bound])

    if plot_output == True:

        for i in range(len(final_data)):
            plt.plot(np.arange(len(final_data[i])), final_data[i], "-")
            plt.plot([all_bounds[0],all_bounds[0]], [-0.1, 0.7], "--", color="green")
            plt.plot([all_bounds[1],all_bounds[1]], [-0.1, 0.7], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    shutil.rmtree("tmp/")


    return final_data, all_bounds[0]

def area_integrator(data, bounds, groups, plot_output=False):

    def linear_baseline(x, m, b):
        return m*x+b

    baseline_xs = []
    baseline_ys = []

    for i in range(len(data)):

        x1 = bounds[0]
        x2 = bounds[1]
        y1 = data[i][bounds[0]]
        y2 = data[i][bounds[1]]
        m = (y2-y1)/(x2-x1)
        b = y1-m*x1

        baseline_x = np.arange(len(data[i][bounds[0]:bounds[1]]))
        baseline_x = baseline_x + bounds[0]

        baseline_y = []

        for i in range(len(baseline_x)):
            y = m*baseline_x[i] + b
            baseline_y.append(y)

        baseline_xs.append(baseline_x)
        baseline_ys.append(baseline_y)

    if plot_output == True:

        for i in range(len(data)):
            plt.plot(np.arange(len(data[i])), data[i], "-")
            plt.plot([bounds[0],bounds[0]], [-0.1, 0.7], "--", color="green")
            plt.plot([bounds[1],bounds[1]], [-0.1, 0.7], "--", color="green")
            plt.plot(baseline_xs[i], baseline_ys[i], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    trunc_data = []

    for i in range(len(data)):
        d = data[i][bounds[0]:bounds[1]]
        d = d - baseline_ys[i]
        trunc_data.append(d)

    areas = []

    for i in range(len(trunc_data)):
        area_trapz = trapz(trunc_data[i])
        areas.append(area_trapz)

    sorted_areas = []

    for i in range(groups):

        index = int(len(data)/groups)
        group = areas[i*index:i*index+index]
        sorted_areas.append(group)

    percentages = []

    for i in range(len(sorted_areas)):
        for j in range(len(sorted_areas[0])):
            if j != 0:
                data_point = 100*sorted_areas[i][j]/sorted_areas[i][0]
                percentages.append(data_point)

    return percentages

def summary_data(datasets, labels, timepoints, colorlist, p0=[100, 0.2], plot_title="", regular_plot=True, df_input_plot=False):

    if regular_plot == True:

        df = pd.DataFrame(datasets, columns=labels)

        plt.figure(figsize=(10,5))
        plt.bar(range(len(df.columns)), df.mean(), align='center', yerr=df.std()/np.sqrt(len(df)), color=colorlist, linewidth=1, edgecolor="black")
        plt.xticks(range(len(df.columns)),labels, fontsize=16)
        plt.plot([-1,int(len(df.columns))], [100, 100], "k--")
        plt.ylim(0,120)
        plt.title(plot_title, fontsize=30)
        plt.ylabel("% remaining after 30 \n minute PK treatment", fontsize=25)
        plt.text(len(labels)-0.2, 110, "n = " + str(len(df)), fontsize=16)
        plt.show()
        None

    # can build plot using df_input_plot if you feed it a dataframe directly via datasets

    if df_input_plot == True:

        df = datasets

        plt.figure(figsize=(10,5))
        plt.bar(range(len(df.columns)), df.mean(), align='center', yerr=df.std()/np.sqrt(len(df)), color=colorlist, linewidth=1, edgecolor="black")
        plt.xticks(range(len(df.columns)), labels, fontsize=16)
        plt.plot([-1,int(len(df.columns))], [100, 100], "k--")
        plt.ylim(0,120)
        plt.title(plot_title, fontsize=30)
        plt.ylabel("% remaining after 30 \n minute PK treatment", fontsize=25)
        plt.text(len(labels)-0.2, 110, "n = " + str(len(df)), fontsize=16)
        plt.show()
        None

    def decay(x, a, k):
        return a * np.exp(-k * x)

    data = np.array(datasets)
    data = data.flatten()

    x = []

    for i in range(len(datasets)):
        x.append(timepoints)

    x = np.array(x)
    x = x.flatten()

    popt, pcov = curve_fit(decay, x, data, p0=p0)
    plt.plot(x,data, ".")
    plt.ylim(-10,150)
    xdata = np.linspace(0,30,50)
    plt.plot(xdata, decay(xdata, *popt))
    plt.show()
    None
    perr = np.sqrt(np.diag(pcov))

    print("intercept = " + str(round(popt[0],4)) + " +/- " + str(round(perr[0],4)))
    print("k = " + str(round(popt[1],4)) + " +/- " + str(round(perr[1],4)) + " per min")

    return df, popt, perr

def half_life_calculator(ks, errs):

    ts = []
    t_errs = []

    for i in range(len(ks)):
        t = (0.693/ks[i])/60
        ts.append(t)

    for i in range(len(errs)):
        d = np.abs(ts[i])*np.sqrt((errs[i]/ks[i])**2)
        t_errs.append(d)

    ts = np.array([ts])
    ts = ts.flatten()
    t_errs = np.array([t_errs])
    t_errs = t_errs.flatten()

    return ts, t_errs

def aggregator(df_list, column=-1):

    means = []
    errors = []

    for i in range(len(df_list)):

        mean = np.mean(df_list[i][df_list[i].columns[column]])
        err = np.std(df_list[i][df_list[i].columns[column]])/np.sqrt(len(df_list[i][df_list[i].columns[column]]))
        means.append(mean)
        errors.append(err)

    return means, errors

def aggregate_plotter(data, errors, labels, colorlist, y_pos, ylabel, xlabel, figname, savefig=False):

    df = pd.DataFrame({"data":data, "errors":errors})
    df['labels'] = labels
    df['colors'] = colorlist

    plt.figure(figsize=(6,4))
    plt.bar(y_pos, df.data, color=df.colors, align='center', yerr=df.errors, width=0.75)
    plt.xticks(y_pos, labels)
    plt.yticks(fontsize=16)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylim(0,105)
    plt.tight_layout()

    if savefig == True:
        plt.savefig(figname, dpi=300)

    None

    return df
