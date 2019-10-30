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
from decimal import Decimal


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

    shutil.rmtree("tmp/", ignore_errors=True)


    return final_data, all_bounds[0]

def area_integrator(data, bounds, groups, plot_output=False, percentages=True):

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

    sorted_areas = [item for sublist in sorted_areas for item in sublist]

    if percentages == True:
        return sorted_areas/sorted_areas[0]
    else:
        return sorted_areas

def summary_data(datasets, timepoints="", output="", p0=[7, 0.2], input_df = False):

    plt.figure(figsize=(2.5,2.5))
    plt.rcParams["font.family"] = "Times New Roman"

    if input_df == True:
        if type(datasets) != pd.core.frame.DataFrame:
            df = pd.read_json(datasets)
            plt.title(datasets.split(".")[0])
            df.to_json(output + ".json")
        else:
            df = datasets
            df.to_json (output + ".json")

    else:

        data = np.array(datasets).flatten()
        time = list(timepoints)*int((len(data)/len(timepoints)))
        time = [int(i) for i in time]
        df = pd.DataFrame({"timepoint":time, "value":data})
        df.to_json(output + ".json")

    def decay(x, a, k):
        return a * np.exp(-k * x)

    popt, pcov = curve_fit(decay, df.timepoint, df.value, p0=p0)
    perr = np.sqrt(np.diag(pcov))

    plt.plot(df.timepoint,df.value, ".")
    plt.ylabel("Normalized \n pixel intensity", fontsize=10)
    plt.xlabel("Time (minutes)", fontsize=10)
    x_decay = np.linspace(0,1000,1000)
    plt.xlim(-1, max(df.timepoint)+5)
    plt.ylim(0,)
    plt.text(0.5,0.5,"k = " + f"{Decimal(str(popt[1])):.2E}" + "\n" + r' $\pm$ ' + f"{Decimal(str(perr[1])):.2E}" + r' min$^{-1}$', fontsize=10)
    plt.plot(x_decay, decay(x_decay, *popt))

    plt.tight_layout()
    plt.savefig(output + "_decay_curve.svg", dpi=100)
    plt.show()
    None

    return popt, perr

def half_life_calculator(ks, errs):

    ts = []
    t_errs = []

    for i in range(len(ks)):
        t = (0.693/ks[i])
        ts.append(t)

    for i in range(len(errs)):
        d = np.sqrt(((0.693/ks[i]**2)**2)*(errs[i]**2))
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
        stderr = np.std(df_list[i][df_list[i].columns[column]])/np.sqrt(len(df_list[i][df_list[i].columns[column]]))
        means.append(mean)
        stderrors.append(stderr)

    return means, stderrors

def aggregate_plotter(data, errors, labels, colorlist, y_pos, ylabel, xlabel, figname, savefig=False):

    df = pd.DataFrame({"data":data, "errors":errors})
    df['labels'] = labelsm63f_hd
    df['colors'] = colorlist

    plt.figure(figsize=(6,4))
    plt.bar(y_pos, df.data, color=df.colors, align='center', yerr=df.errors, width=0.75)
    plt.xticks(range(len()))
    plt.yticks(fontsize=16)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylim(0,105)
    plt.tight_layout()

    if savefig == True:
        plt.savefig(figname, dpi=300)

    None

    return df
