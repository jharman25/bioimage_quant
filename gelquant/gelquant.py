from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os
import shutil
from natsort import natsorted, ns
from scipy.integrate import trapz
from scipy.stats import norm

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
            plt.plot([all_bounds[i][0],all_bounds[i][0]], [-0.1, 0.7], "--", color="green")
            plt.plot([all_bounds[i][1],all_bounds[i][1]], [-0.1, 0.7], "--", color="green")
            plt.ylim(-0.1, 0.7)
            plt.show()

    shutil.rmtree("tmp/")

    print("Done.")

    return final_data, all_bounds

def area_integrator(data, bounds, groups):

    areas = []

    for i in range(len(data)):
        area_trapz = trapz(data[i][bounds[i][0]:bounds[i][1]])
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

def summary_plotter(dataset, labels, colorlist):

    plt.figure(figsize=(10,5))
    df = pd.DataFrame(datasets, columns=labels)
    plt.bar(range(len(df.columns)), df.mean(), align='center', yerr=df.std(), color=colorlist)
    plt.xticks(range(len(df.columns)), df.columns, rotation=70, fontsize=14)
    plt.plot([-1,int(len(df.columns))], [100, 100], "k--")
    plt.ylim(0,120)
    plt.title("Proteolytic susceptibility of calgranulin complexes", fontsize=20)
    plt.ylabel("% remaining after PK treatment", fontsize=16)
    plt.xlabel("Calgranulin complex", fontsize=16)
    None

    return df
