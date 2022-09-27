"""
@Filename: tropSF6rigby.py

@Author: Thomas Wagenhäuser, IAU
@Date:   2022-02-11T07:47:42+01:00
@Email:  wagenhaeuser@iau.uni-frankfurt.de

"""

import pandas as pd
import numpy as np
# import matplotlib
#
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

# plt.ion()
# plt.rcParams.update({"font.size": 20})

from scipy import interpolate
import xarray
from pathlib import Path

# %% ##########################################
# check out Rigby's MOZART nc files
###############################################

ds = xarray.open_dataset(
    r"C:\acp-10-10305-2010-supplement\RIGBY_2010_SF6_MOLE_FRACTION_1970_2008.nc"
)
ds
sf6 = ds.SF6


# %%###########################################
# calculate latitude weights for accurate averaging!
###############################################
weights_lat = np.cos(np.deg2rad(sf6.latitude))
weights_lat.name = "latitude_weights"
# weights_lat
# sf6.level
weights_level = sf6.level
weights_level.name = "level_weights"
weights_latlevel = weights_lat * weights_level
weights_latlevel.name = "latitude_level_weights"

# plot weights_lat
plt.figure()
weights_lat.plot()
weights_level.plot()

# example calculation:
sf6_weighted_latmean = sf6.weighted(weights_lat).mean(["latitude", "longitude"])
sf6_weighted_latmean


# %%############################################
# define functions for selecting and averaging subsets of sf6 fields
################################################
def choose_weights_for_dim(dim):
    if dim is None:
        dim = ""
    _weights = weights_level / weights_level  # should be a DataArray of ones
    _weights.name = "weight_equal"
    lat = "latitude" in dim
    lev = "level" in dim
    latlev = lat and lev
    if lat:
        _weights = weights_lat
    if lev:
        _weights = weights_level
    if latlev:
        _weights = weights_latlevel
    return _weights


def _get_slice_or_nearest(sf6, dim, diminput):
    if diminput is None:
        diminput = (None,)
    if isinstance(diminput, int) or isinstance(diminput, float):
        sf6_subset = sf6.sel({dim: diminput}, method="nearest")
    else:
        sf6_subset = sf6.sel({dim: slice(*diminput)})
    return sf6_subset


def sel_subset(sf6, latitude=None, level=None, longitude=None, time=None):
    dim_dict = {
        "latitude": latitude,
        "level": level,
        "longitude": longitude,
        "time": time,
    }
    sf6_subset = sf6
    for dim, diminput in dim_dict.items():
        sf6_subset = _get_slice_or_nearest(sf6_subset, dim, diminput)
    return sf6_subset


def get_dim_mean(sf6, dim=None):
    _weights = choose_weights_for_dim(dim)
    sf6_m = sf6.weighted(_weights).mean(dim)
    return sf6_m


def sel_get_dim_mean(
    sf6, dim=None, latitude=None, level=None, longitude=None, time=None
):
    sf6_subset = sel_subset(sf6, latitude, level, longitude, time)
    mean = get_dim_mean(sf6_subset, dim)
    return mean


# %%#######################################
# define functions to get standard deviation for subsets of sf6
###########################################
def get_dim_std(sf6, dim=None):
    _weights = choose_weights_for_dim(dim)
    std = sf6.weighted(_weights).std(dim)
    return std


def sel_get_dim_std(
    sf6, dim=None, latitude=None, level=None, longitude=None, time=None
):
    sf6_subset = sel_subset(sf6, latitude, level, longitude, time)
    std = get_dim_std(sf6_subset, dim)
    return std


def get_dim_mean_std(sf6, dim=None):
    mean = get_dim_mean(sf6, dim)
    std = get_dim_std(sf6, dim)
    return mean, std


def sel_get_dim_mean_std(
    sf6, dim=None, latitude=None, level=None, longitude=None, time=None
):
    sf6_subset = sel_subset(sf6, latitude, level, longitude, time)
    mean, std = get_dim_mean_std(sf6_subset, dim)
    return mean, std


# %%#######################################
# define functions to calculate time shift relative to sf6_ref
###########################################
def calculate_shift_from_ref(sf6, c2y_ref, time_ref):
    yfc = sf6.copy()  # yfc = Year From Concentration
    yfc.name = "yfc"
    yfc.values = c2y_ref(sf6)
    # shift = yfc - time_ref  # prior to 2022-03-01
    shift = time_ref - yfc
    return shift


def sel_calc_shift_and_average_for_dim(
    sf6,
    c2y_ref,
    time_ref,
    dim=None,
    latitude=None,
    level=None,
    longitude=None,
    time=None,
):
    sf6_subset = sel_subset(sf6, latitude, level, longitude, time)
    shift = calculate_shift_from_ref(sf6_subset, c2y_ref, time_ref)
    mean, std = get_dim_mean_std(shift, dim)
    return mean, std


# %%#########################################
# define functions to plot time shift time series
#############################################
def add_errorbarplot(ax, shift, std_y, title=None, label=None):
    ax.errorbar(
        x=shift.time,
        y=shift.values,
        yerr=std_y,
        marker="o",
        ls="",
        capsize=3,
        label=label,
    )
    if title:
        ax.set_title(title)
    ax.set_ylabel("time shift / years")
    ax.set_xlabel("date")
    return ax


def add_errorbarplot_tsmean(ax, meanshift, std_y, title=None, label=None, hemi=None):
    # x = {"SH": 0, "TR": 1, "SH": 2}
    if hemi is None:
        hemi = np.arange(len(np.asarray(meanshift)))
    ax.errorbar(
        x=hemi, y=meanshift, yerr=std_y, marker="o", ls="", capsize=3, label=label
    )
    if title:
        ax.set_title(title)
    ax.set_ylabel("time shift / years")
    ax.set_xlabel("location")
    return ax


# %%#########################################
# define functions to calculate and plot time shift time series
#############################################
# sel first, then shift, then average
def sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref,
    dim=None,
    latitude=None,
    level=None,
    longitude=None,
    time=None,
    title=None,
    label=None,
    ax=None,
    hemi=None,
):
    shift_mean, shift_std = sel_calc_shift_and_average_for_dim(
        sf6, c2y_ref, time_ref, dim, latitude, level, longitude, time
    )
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if len(np.asarray(shift_mean).flatten()) == 1:
        ax = add_errorbarplot_tsmean(ax, shift_mean, shift_std, title, label, hemi)
    else:
        ax = add_errorbarplot(ax, shift_mean, shift_std, title, label)
    return fig, ax, shift_mean, shift_std


# %%###########################################
# Monte Carlo approach to test for trend
###############################################
def generate_MC_parameters(
    dimmeans, dimstds, N=None, seed=None,
):
    rng = np.random.default_rng(seed=seed)
    MC_per_year = []
    for _mean, _std in zip(dimmeans, dimstds):
        _MC_per_year = rng.normal(_mean, _std, size=N)
        MC_per_year.append(_MC_per_year)
    return np.vstack(MC_per_year)


def MC_polyfit(years, MC_per_year, deg=1):
    coef = np.polynomial.polynomial.polyfit(
        np.asarray(years), np.asarray(MC_per_year), deg=deg
    )
    mean_slope = coef[-1, :].mean()
    std_slope = coef[-1, :].std()
    return mean_slope, std_slope


def MC_test_trend(dimmeans, dimstds, years, N=10, seed=None):
    MC_per_year = generate_MC_parameters(dimmeans, dimstds, N=N, seed=seed)
    mean_slope, std_slope = MC_polyfit(years, MC_per_year, deg=1)
    return mean_slope, std_slope


# %%############################################
# calculate reference ground time series
################################################
sf6_ref = sel_get_dim_mean(
    sf6,
    dim=["latitude", "longitude"],
    latitude=(-30, 30),
    level=1000,
    longitude=None,
    time=None,
)


# get interpolation function
c2y_ref = interpolate.interp1d(
    sf6_ref.values, sf6_ref.time, fill_value="extrapolate", kind="quadratic"
)


plt.figure()
sf6_ref.plot()


# %%############################################
# plot time shift time series
################################################
# create NH plot
(fig1, ax1, shift_mean1, shift_std1,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude"],
    latitude=(30, 90),
    level=300,
    longitude=None,
    time=None,
    title="NH time shift to TR ground",
    label=" @ 300 hPa",
    ax=None,
)

# add NH200500 plot
(fig1, ax1, shift_mean1b, shift_std1b,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level"],
    latitude=(30, 90),
    level=(200, 500),
    longitude=None,
    time=None,
    title=None,
    label="@ 200-500 hPa",
    ax=ax1,
)

ax1.legend()

shift1b_slope_mean, shift1b_slope_std = MC_test_trend(
    shift_mean1b[1:], shift_std1b[1:], shift_mean1b.time[1:], N=10000
)
print(shift1b_slope_mean)
print(shift1b_slope_std)


# %%############################################
# create TR plot
(fig2, ax2, shift_mean2, shift_std2,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude"],
    latitude=(-30, 30),
    level=150,
    longitude=None,
    time=None,
    title="TR time shift to TR ground",
    label=" @ 150 hPa",
    ax=None,
)

# add TR100300 plot
(fig2, ax2, shift_mean2b, shift_std2b,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level"],
    latitude=(-30, 30),
    level=(100, 300),
    longitude=None,
    time=None,
    title=None,
    label="@ 100-300 hPa",
    ax=ax2,
)
ax2.legend()

shift2b_slope_mean, shift2b_slope_std = MC_test_trend(
    shift_mean2b[1:], shift_std2b[1:], shift_mean2b.time[1:], N=10000
)
print(shift2b_slope_mean)
print(shift2b_slope_std)


# %%############################################
# create SH plot
(fig3, ax3, shift_mean3, shift_std3,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude"],
    latitude=(-90, -30),
    level=300,
    longitude=None,
    time=None,
    title="SH time shift to TR ground",
    label=" @ 300 hPa",
    ax=None,
)

# add SH200500 plot
(fig3, ax3, shift_mean3b, shift_std3b,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level"],
    latitude=(-90, -30),
    level=(200, 500),
    longitude=None,
    time=None,
    title=None,
    label="@ 200-500 hPa",
    ax=ax3,
)
ax3.legend()

shift3b_slope_mean, shift3b_slope_std = MC_test_trend(
    shift_mean3b[1:], shift_std3b[1:], shift_mean3b.time[1:], N=10000
)
print(shift3b_slope_mean)
print(shift3b_slope_std)
print(shift3b_slope_mean / shift3b_slope_std)


# %%############################################
# plot time shift mean
################################################
# create NH plot
(
    fig1,
    ax1,
    shift_mean1_all,
    shift_std1_all,
) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "time"],
    latitude=(30, 90),
    level=300,
    longitude=None,
    time=(1973, 2008),
    title="NH mean time shift to TR ground",
    label=" @ 300 hPa",
    ax=None,
    hemi="NH",
)


# add NH200500 plot
(
    fig1,
    ax1,
    shift_mean1b_all,
    shift_std1b_all,
) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level", "time"],
    latitude=(30, 90),
    level=(200, 500),
    longitude=None,
    time=(1973, 2008),
    title=None,
    label="@ 200-500 hPa",
    ax=ax1,
    hemi="NH",
)
ax1.legend()

# %%############################################
# create TR plot
(
    fig2,
    ax2,
    shift_mean2_all,
    shift_std2_all,
) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "time"],
    latitude=(-30, 30),
    level=150,
    longitude=None,
    time=(1973, 2008),
    title="Mean time shift to TR ground",
    label=" @ 150 hPa",
    ax=ax1,
    hemi="TR",
)

# add TR100300 plot
(
    fig2,
    ax2,
    shift_mean2b_all,
    shift_std2b_all,
) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level", "time"],
    latitude=(-30, 30),
    level=(100, 300),
    longitude=None,
    time=(1973, 2008),
    title=None,
    label="@ 100-300 hPa",
    ax=ax1,
    hemi="TR",
)
ax1.legend()


# %%############################################
# create SH plot
(
    fig3,
    ax3,
    shift_mean3_all,
    shift_std3_all,
) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "time"],
    latitude=(-90, -30),
    level=300,
    longitude=None,
    time=(1973, 2008),
    title=None,
    label=" @ 300 hPa",
    ax=ax1,
    hemi="SH",
)

# add SH200500 plot
(
    fig3,
    ax3,
    shift_mean3b_all,
    shift_std3b_all,
) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level", "time"],
    latitude=(-90, -30),
    level=(200, 500),
    longitude=None,
    time=(1973, 2008),
    title=None,
    label="@ 200-500 hPa",
    ax=ax1,
    hemi="SH",
)
ax1.legend()


print(shift_mean1b_all.values)
print(shift_mean2b_all.values)
print(shift_mean3b_all.values)

print(shift_std1b_all.values)
print(shift_std2b_all.values)
print(shift_std3b_all.values)


# %%########################################
# save plots
############################################
def save_and_close_all_open_figs(folder, save=True, close=True):
    if save:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        for n, fig in enumerate(figs):
            _spath = Path(folder) / "fig{:02}.png".format(n)
            fig.savefig(_spath, dpi=300)
    if close:
        plt.close("all")


save_plot = True
close_all = True
spath = Path(r"C:\WiMi_ab_Maerz2019\Southtrac\Age-Paper\figures\time_shifts")
save_and_close_all_open_figs(spath, save=save_plot, close=close_all)


# %%########################################
# save values to .csv file
############################################
fpath = Path(r"C:\WiMi_ab_Maerz2019\Southtrac\Age-Paper\data\time_shifts")
filename = "SF6_time_shift_from_MOZART.csv"
# generate header information
param_names = [
    "NH_shift_mean",
    "NH_shift_std",
    "TR_shift_mean",
    "TR_shift_std",
    "SH_shift_mean",
    "SH_shift_std",
]

header = [
    filename,
    "Mean SF6 time series shifts for target regions relative to tropical ground\n",
    "The mean time shifts were calculated as follows:\n"
    "1) Calculated mean tropical ground SF6 time series between -30° to 30° N",
    "2) SF6 from target region was interpolated to tropical ground time.",
    "3) Time shifts were calculated by subtracting the result from 2) from the tropical ground time.",
    "4) Mean values and standard deviations were calculated, weighted by latitude and pressure.\n",
    "Time shifts were calculated for the following regions:",
    "Northern Hemisphere (NH): 30 to 90° N, 200 to 500 hPa",
    "Tropics (TR): -30 to 30° N, 100 to 300 hPa",
    "Southern Hemisphere (SH): -90 to -30° N, 200 to 500 hPa\n",
    "Time shifts are given in years.",
    "Positive time shifts indicate, that the corresponding region lags behind tropical ground SF6 values.",
    "Negative time shifts indicate, that the corresponding region precedes tropical ground SF6 values.\n",
    "###########################################################################",
    "All calculations are based on SF6 data given in the file RIGBY_SF6_MOLE_FRACTIONS_1970_2008.nc",
    "This is a NetCDF file containing annual mean optimized three-dimensional mole fractions output from MOZART for 1970 - 2008.",
    "The file is part of the Supplementary Material of the following publication:"
    "Rigby, M., Mühle, J., Miller, B. R., Prinn, R. G., Krummel, P. B., Steele, L. P., Fraser, P. J., Salameh, P. K., Harth, C. M., Weiss, R. F., Greally, B. R., O'Doherty, S., Simmonds, P. G., Vollmer, M. K., Reimann, S., Kim, J., Kim, K. R., Wang, H. J., Dlugokencky, E. J., Dutton, G. S., Hall, B. D., and Elkins, J. W.: History of atmospheric SF6 from 1973 to 2008, Atmos. Chem. Phys. Discuss., 10, 13519-13555",
    "###########################################################################\n",
    ",".join([param for param in param_names]),
]
header = "\n".join(header)

# prepare data
# data = pd.DataFrame(
#     {
#         "NH": [shift_mean1b.values, shift_std1b.values],
#         "TR": [shift_mean2b.values, shift_std2b.values],
#         "SH": [shift_mean3b.values, shift_std3b.values],
#     },
#     index=["weighted_mean", "weighted_std"],
# )

data = np.array(
    [
        shift_mean1b.values,
        shift_std1b.values,
        shift_mean2b.values,
        shift_std2b.values,
        shift_mean3b.values,
        shift_std3b.values,
    ]
)

# with open(fpath / filename, "w") as f:
#     f.write(header)
# data.to_csv(fpath/filename, float_format="%.3f", sep=",", mode="a")
np.savetxt(
    fpath / filename, data.reshape(1, -1), fmt="%.3f", delimiter=",", header=header,
)


# %%##################################################
# create plots for paper
######################################################
(fig1, ax1, shift_mean1b, shift_std1b,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level"],
    latitude=(30, 90),
    level=(200, 500),
    longitude=None,
    time=(1973, 2008),
    title="time shifts to tropical ground",
    label="NH @ 200-500 hPa",
    ax=None,
)

(fig1, ax1, shift_mean2b, shift_std2b,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level"],
    latitude=(-30, 30),
    level=(100, 300),
    longitude=None,
    time=(1973, 2008),
    title=None,
    label="TR @ 100-300 hPa",
    ax=ax1,
)

(fig1, ax1, shift_mean3b, shift_std3b,) = sel_calc_shift_and_average_for_dim_and_plot(
    sf6,
    c2y_ref,
    time_ref=sf6_ref.time,
    dim=["latitude", "longitude", "level"],
    latitude=(-90, -30),
    level=(200, 500),
    longitude=None,
    time=(1973, 2008),
    title=None,
    label="SH @ 200-500 hPa",
    ax=ax1,
)

ax1.legend()
fig1.savefig(spath / "time_series.png", dpi=300)


shift1b_slope_mean, shift1b_slope_std = MC_test_trend(
    shift_mean1b, shift_std1b, shift_mean1b.time, N=10000
)
print(shift1b_slope_mean)
print(shift1b_slope_std)
print(shift1b_slope_mean / shift1b_slope_std)
print(shift1b_slope_std / shift1b_slope_mean)

shift2b_slope_mean, shift2b_slope_std = MC_test_trend(
    shift_mean2b, shift_std2b, shift_mean2b.time, N=10000
)
print(shift2b_slope_mean)
print(shift2b_slope_std)
print(shift2b_slope_mean / shift2b_slope_std)
print(shift2b_slope_std / shift2b_slope_mean)

shift3b_slope_mean, shift3b_slope_std = MC_test_trend(
    shift_mean3b, shift_std3b, shift_mean3b.time, N=10000
)
print(shift3b_slope_mean)
print(shift3b_slope_std)
print(shift3b_slope_mean / shift3b_slope_std)
print(shift3b_slope_std / shift3b_slope_mean)
