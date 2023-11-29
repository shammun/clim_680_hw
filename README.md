# The Effect of Indian Ocean Dipole 🌊 on Temperature ☀️ in Asia 

**Shammunul Islam (sislam27@mason.edu)** 👋

## ✨ Intro 

Due to climate variability and climate change, the world 🌎 is continuously affected by different extreme events, especially associated with high temperature. Asia is affected by many calamities caused by high temperature driven events. Indian Ocean 🌊 plays a critical role in the atmospheric dynamics of this region, particularly, Indian Ocean Dipole (IOD) 🌊, an oscillation of temperature in Indian Ocean equivalent to ENSO in central and eastern tropical Pacific Ocean. In this project, it is investigated whether there is any relationship between different phases of IOD 🌊, which areas have these relationships and whether these are statistically significant.

For doing this study, APHRODITE Water Resources project data for daily mean temperature ☀️ is chosen as they have daily temperature data from 1962 upto 2015 in a gridded format with a high spatial resolution of 0.5 degree by 0.25 degree. For sea surface temperature, NOAA OISST V2 SST data is used as it has good spatial and temporal coverage.

There are mainly 3 notebooks [HW1_New.ipynb](https://github.com/shammun/clim_680_hw/blob/main/HW1_New.ipynb), [HW2_New.ipynb](https://github.com/shammun/clim_680_hw/blob/main/HW2_New.ipynb), and [HW3_New.ipynb](https://github.com/shammun/clim_680_hw/blob/main/HW3_New.ipynb) that show you step by step process with Python code on how to analyze temperature and sea surface temperature data using XArray, Dask and many more!! This repository contains a collection of insightful Jupyter Notebooks and a dedicated GitHub Page built in requirement for the class project of 680: Climate Data, a course required for PhD in CLimate Dynamics, at George Mason Universityy.

One caveat though 🙃, as the data size is huge, you will have to download the data by yourself in your local computer before you can run the code. 

Ohh, by the way, look at this Github page 📖 [https://shammun.github.io/clim_680_hw/](https://shammun.github.io/clim_680_hw/) associated with the repository that walks you through the codes and that also discusses the results. 

## 🌟 Key Features:

Interactive Notebooks
Detailed Analysis with Python Code 🐍
GitHub Page 📖 [Github Page](https://shammun.github.io/clim_680_hw/)

Some examples:

<button onclick="toggleVisibility('image4', 'code4')" style="background-color: #0066cc; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
    Toggle between image and code
</button>
<!-- Image -->
<img src="jjas_temperature_anomaly.gif" id="image4" style="display:block;">



<pre id="code4" style="display:none; background-color: #f7f7f7; border-left: 5px solid #0066cc; padding: 10px; margin: 10px 0; overflow: auto; font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6;">
  <code>
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker

def load_dataset(year):
    filename = f"APHRO_MA_TAVE_025deg_V1808.{year}.nc.nc"
    ds = xr.open_dataset(filename)
    return ds

def calculate_jjas_anomaly(year, long_term_mean):
    jjas_data = load_dataset(year)['tave'].sel(time=load_dataset(year)['time.month'].isin([6, 7, 8, 9])).mean(dim='time')
    anomaly = jjas_data - long_term_mean
    return anomaly

def plot_data(data, ax):
    lons, lats = np.meshgrid(data['lon'], data['lat'])
    # Adjust the level_interval for finer gradations
    level_interval = 0.2  # Smaller interval for more detailed color gradations
    levels = np.linspace(data.min(), data.max(), num=int((data.max() - data.min()) / level_interval))
    cs = ax.contourf(lons, lats, data, levels=levels, transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both')
    return cs

# Compute the long-term JJAS mean (1961-2015)
long_term_mean = xr.concat([load_dataset(year)['tave'].sel(time=load_dataset(year)['time.month'].isin([6, 7, 8, 9])) for year in range(1961, 2016)], dim='time').mean(dim='time')

# Create the initial figure and axis with adjusted vertical size
fig, ax = plt.subplots(figsize=(11, 10), subplot_kw={'projection': ccrs.PlateCarree()})
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Load the first year's data to initialize the colorbar
initial_data = calculate_jjas_anomaly(1961, long_term_mean)
cs = plot_data(initial_data, ax)

# Adjust the colorbar to align with the map's extent
cbar_ax = fig.add_axes([0.125, 0.08, 0.775, 0.03])  # Slightly raise the colorbar
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='Temperature Anomaly (Celsius)')

def update_plot(year):
    ax.clear()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    data = calculate_jjas_anomaly(year, long_term_mean)
    cs = plot_data(data, ax)
    ax.set_title(f"JJAS Temperature Anomaly for {year}", fontsize=14, pad=20)
    
    # Set dynamic longitude and latitude labels based on the data
    lon_range = np.arange(np.floor(data['lon'].min()), np.ceil(data['lon'].max()) + 1, 10)
    lat_range = np.arange(np.floor(data['lat'].min()), np.ceil(data['lat'].max()) + 1, 10)
    ax.set_xticks(lon_range, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_range, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    return cs

# Create the animation
years = range(1961, 2016)
ani = animation.FuncAnimation(fig, update_plot, frames=years, repeat=True, blit=False)

# Save the animation as a .gif
ani.save('jjas_temperature_anomaly.gif', writer='imagemagick', fps=1.8)

plt.show()
  </code>
</pre>


## 🚀 Getting Started
These instructions will get you a copy of the project up and running on your local machine to be able to run the notebooks.

## 🧰 Prerequisites
What things you need to install the software and how to install them:

* [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* [Git](https://git-scm.com/downloads)

## ⚙️ Installation
A step-by-step series of examples that tell you how to get a development environment running:

1. 💾 Clone the Repository: 

`git clone https://github.com/shammun/clim_680_hw`

2. 🧭 Navigate to the Repository:

`cd clim_680_hw`

3. 🌱 Create the Conda Environment:

`conda env create -f environment.yml`

4. 🔧 Activate the Environment:

`conda activate [Your Environment Name]`

5. 📓 Launch Jupyter Notebook:

`jupyter notebook`

6. View 👀 GitHub 🐙 Page 📄:

* Visit [https://shammun.github.io/clim_680_hw/] to explore the GitHub Page associated with this project.

## 📚 Documentation
For more detailed information about this class project, refer to the following resources:

Notebook Descriptions: Each notebook contains details on what to do and how to do.
GitHub Page: For a more interactive experience and ready-made visualization.

## 📩 Contact
Shammunul Islam - [sha_is13@yahoo.com, shais13irs@gmail.com, si2267@gmu.edu]

Project Link: [https://shammun.github.io/clim_680_hw/]


<script>
  function toggleVisibility(imageId, codeId) {
    var image = document.getElementById(imageId);
    var code = document.getElementById(codeId);

    if (image.style.display === "none") {
      image.style.display = "block";
      code.style.display = "none";
    } else {
      image.style.display = "none";
      code.style.display = "block";
    }
  }
</script>

<script>
  function toggleVideoAndCode(videoId, codeId) {
    var video = document.getElementById(videoId);
    var code = document.getElementById(codeId);

    if (video.style.display === "none") {
      video.style.display = "block";
      code.style.display = "none";
    } else {
      video.style.display = "none";
      code.style.display = "block";
    }
  }
</script>
