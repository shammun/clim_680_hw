# The Effect of Indian Ocean Dipole on Temperature in Asia 

### Introduction 

Due to climate variability and climate change, the world is continuously affected by different extreme events, especially associated with high temperature. Asia is affected by many calamities caused by high temperature driven events. Indian Ocean plays a critical role in the atmospheric dynamics of this region and particularly, Indian Ocean Dipole (IOD), an oscillation of temperature in Indian Ocean equivalent to ENSO in central and eastern tropical Pacific Ocean. In this project, it is investigated whether there is any relationship between different phases of IOD, which areas have these relationships and whether these are statistically significant.

For doing this study, APHRODITE Water Resources project data for daily mean temperature is chosen as they have daily temperature data from 1962 upto 2015 in a gridded format with a high spatial resolution of 0.5 degree by 0.25 degree. For sea surface temperature, NOAA OISST V2 SST data is used as it has good spatial and temporal coverage.

### Data 

#### Temperature Data -- The APHRODITE Water Resources project

The APHRODITE Water Resources project is working on a series of precipitation products and this project completed its first phase in 2010 and embarked on its second phase in 2016, partnering with Hirosaki University, Kyoto University, and Chiba University. 

Daily mean temperature values are provided by their AphroTemp_V1808 at 0.50 and 0.25 degree grid resolutions. These readings are produced by extrapolating information from meteorological stations dispersed throughout the targeted area. After the release of AphroTemp_V1204R1, surface data from nations including India, Bhutan, Thailand, and Myanmar was included. The accuracy of the temperature readings in South Asia in the most recent version has been improved by this inclusion, along with updated interpolation methods and climatic data. For a better understanding of this dataset, you can refer to this [documentation](http://aphrodite.st.hirosaki-u.ac.jp/product/APHRO_V1808_TEMP/AphroTemp_V1808_readme.txt) 

##### The major characteristics of the dataset

- Spatial and Temporal Coverage

   * Spatial coverage      :  (MA) 60.0E - 150.0E, 15.0S - 55.0N
   * Spatial resolution    :  0.5 degree and 0.25 degree latitude/longitude
   * Temporal coverage     :  1961-2015
   * Temporal resolution   :  Daily

- Units
   * Daily mean temperature :  degC
   * Ratio of 0.05 grid box containing station(s) :  %

- Missing Code

   * Daily mean temperature :  -99.9
   * Ratio of 0.05 grid box containing station(s) :  -99.9

#### NOAA OI SST V2 High Resolution Dataset

This gridded dataset NOAA OI SST V2 High Resolution Dataset can be found at [this link](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html).

- Spatial and Temporal Coverage

   * Daily values from 1981/09 to 2023/11
   * 0.25 degree latitude x 0.25 degree longitude global grid (1440x720)
   * 89.875S - 89.875N,0.125E to 359.875E

- Missing Data
   * Missing data is flagged with a value of -9.96921e+36f.

### Code description or data analysis

#### Temperature analysis
First of all, air temperature (in Celsius) is plotted over Asia using different Python libraries, including Cartopy and Matplotlib. Detailed step-by-step methods with code can be found in this [notebook](HW1_New.ipynb). But, the code for each of the figures displayed here can also be seen by clicking on the button placed above each figure. 

<!-- Toggle Button -->
<button onclick="toggleVisibility('image1', 'code1')" style="background-color: #0066cc; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
    Toggle between image and code
</button>
<!-- Image -->
<img src="air_temp_1961.png" id="image1" style="display:block;">

<!-- Code Block (initially hidden) -->
<pre id="code1" style="display:none; background-color: #f7f7f7; border-left: 5px solid #0066cc; padding: 10px; margin: 10px 0; overflow: auto; font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6;">
  <code>
    // Your code here
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature


# Make the figure larger
fig = plt.figure(figsize=(11, 8.5))

# Set the axes using the specified map projection
# Adjust the position of the map to leave space for the colorbar
ax = plt.axes([0.05, 0.2, 0.9, 0.7], projection=ccrs.PlateCarree())

# Add cyclic point to data
data = ds_mean_1961['tave']
data, lons = add_cyclic_point(data, coord=ds_1961['lon'])

# Define levels for finer intervals, ignoring NaNs
level_interval = 2 
levels = np.arange(np.nanmin(data), np.nanmax(data) + level_interval, level_interval)

# Make a filled contour plot with specified levels
cs = ax.contourf(lons, ds_1961['lat'], data, levels=levels,
                 transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both')

# Add coastlines
ax.coastlines()
# Add country boundaries
ax.add_feature(cfeature.BORDERS, edgecolor='black')

# Define the xticks for longitude
lon_range = np.arange(np.floor(ds_1961['lon'].min()), np.ceil(ds_1961['lon'].max()) + 1, 20)
ax.set_xticks(lon_range, crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
lat_range = np.arange(np.floor(ds_1961['lat'].min()), np.ceil(ds_1961['lat'].max()) + 1, 10)
ax.set_yticks(lat_range, crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter) 

# Define the longitude and latitude range
ax.set_extent([ds_1961['lon'].min(), ds_1961['lon'].max(), ds_1961['lat'].min(), ds_1961['lat'].max()])

# Add title
plt.title("Air Temperature (Celsius) in 1961")

# Create a new axes for the colorbar just below the map
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])
cbar = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='Surface Air Temperature (Celsius)')

# Save and show the plot
plt.savefig('air_temp_1961.png', dpi=300, bbox_inches='tight')
plt.show()
  </code>
</pre>


Now, let's see if the temperature changed in 2015 compared to 1961 by plotting the temperature anomaly.

<button onclick="toggleVisibility('image2', 'code2')" style="background-color: #0066cc; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
    Toggle between image and code
</button>
<!-- Image -->
<img src="air_temp_anomaly_2015_1961.png" id="image2" style="display:block;">


<pre id="code2" style="display:none; background-color: #f7f7f7; border-left: 5px solid #0066cc; padding: 10px; margin: 10px 0; overflow: auto; font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6;">
  <code>
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature

fname = 'APHRO_MA_TAVE_025deg_V1808.2015.nc.nc'
ds_2015 = xr.open_dataset(fname)
ds_mean_2015=ds_2015.mean(dim='time')
fname = 'APHRO_MA_TAVE_025deg_V1808.1961.nc.nc'
ds_1961 = xr.open_dataset(fname)
ds_mean_1961=ds_1961.mean(dim='time')
anomaly = ds_mean_2015 - ds_mean_1961

# Make the figure larger
fig = plt.figure(figsize=(11, 8.5))

# Set the axes using the specified map projection
ax = plt.axes(projection=ccrs.PlateCarree(), position=[0.05, 0.15, 0.9, 0.7])  # Adjust the position of the map

# Add cyclic point to data
data = anomaly['tave']
data, lons = add_cyclic_point(data, coord=ds_2015['lon'])

# Define levels for finer intervals, considering a more detailed range
min_val = np.nanmin(data)
max_val = np.nanmax(data)
level_interval = 0.5  # Smaller interval for more granularity
levels = np.arange(min_val, max_val + level_interval, level_interval)

# Make a filled contour plot with specified levels
cs = ax.contourf(lons, ds_2015['lat'], data, levels=levels,
                 transform=ccrs.PlateCarree(), cmap='coolwarm', extend='both')

# Add coastlines
ax.coastlines()
# Country boundaries
ax.add_feature(cfeature.BORDERS, edgecolor='black')

# Define the xticks for longitude
lon_range = np.arange(np.floor(ds_2015['lon'].min()), np.ceil(ds_2015['lon'].max()) + 1, 20)
ax.set_xticks(lon_range, crs=ccrs.PlateCarree())
lon_formatter = cticker.LongitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)

# Define the yticks for latitude
lat_range = np.arange(np.floor(ds_2015['lat'].min()), np.ceil(ds_2015['lat'].max()) + 1, 10)
ax.set_yticks(lat_range, crs=ccrs.PlateCarree())
lat_formatter = cticker.LatitudeFormatter()
ax.yaxis.set_major_formatter(lat_formatter) 

# Define the longitude and latitude range
ax.set_extent([ds_2015['lon'].min(), ds_2015['lon'].max(), ds_2015['lat'].min(), ds_2015['lat'].max()])

# Add title with adjusted position
plt.title("Air temperature anomaly between 2015 and 1961 (Celsius)", pad=20)

# Reposition the colorbar to increase the gap from the map
cbar_ax = fig.add_axes([0.15, 0.07, 0.7, 0.03])  # Adjusted to increase the gap
cbar = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='Air Temperature difference (Celsius)')

plt.savefig('air_temp_anomaly_2015_1961.png', dpi=300, bbox_inches='tight')
plt.show()
  </code>
</pre>

##### June, July, August, September (JJAS) temperature

This is normally the summer season and here, I have investigated whether temperature increased in Asia in this season. Let's look at the temperature evolution over the period from 1961 to 2015.

<button onclick="toggleVisibility('image3', 'code3')" style="background-color: #0066cc; color: white; border: none; padding: 10px 20px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer; border-radius: 8px;">
    Toggle between image and code
</button>
<!-- Image -->
<img src="average_temp_JJAS_over_years.png" id="image3" style="display:block;">


<pre id="code3" style="display:none; background-color: #f7f7f7; border-left: 5px solid #0066cc; padding: 10px; margin: 10px 0; overflow: auto; font-family: 'Courier New', Courier, monospace; font-size: 14px; line-height: 1.6;">
  <code>
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the style for the plot
plt.style.use('seaborn-whitegrid')
sns.set_context('talk')

# Assuming ds is your loaded xarray dataset
# Select the temperature variable (tave)
tave = ds['tave']

# Filter the dataset to include only JJAS (months 6, 7, 8, and 9)
tave_jjas = tave.sel(time=tave['time'].dt.month.isin([6, 7, 8, 9]))

# Calculate the mean for each JJAS across latitude and longitude dimensions
jjas_mean = tave_jjas.mean(dim=['lat', 'lon'])

# Group by year and then calculate the mean for each year
yearly_avg = jjas_mean.groupby('time.year').mean()

# Convert to a pandas DataFrame for plotting
yearly_avg_df = yearly_avg.to_dataframe().reset_index()

# Plotting
plt.figure(figsize=(15, 8))

plt.plot(yearly_avg_df['year'], yearly_avg_df['tave'], marker='o', linestyle='-', color='royalblue')

plt.xlabel('Year')
plt.ylabel('Average Temperature (JJAS)')
plt.title('Average Temperature for JJAS Over Years in Asia')

# Adding a grid for better readability
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Enhance aesthetics
plt.xticks(rotation=45)
plt.tight_layout()

# Adding minor ticks
plt.minorticks_on()

# Adding custom text
credit_text = "Prepared by:\nShammunul Islam\nData Source: APHRODITE Daily Mean Temperature Dataset"
plt.text(0.65, 0.05, credit_text, transform=plt.gca().transAxes, fontsize=10, 
         verticalalignment='bottom', horizontalalignment='left', 
         bbox=dict(facecolor='white', alpha=0.5))

# Adjust the padding of the plot, if necessary
plt.subplots_adjust(bottom=0.15, right=0.85)

# Save and show the plot
plt.savefig('average_temp_JJAS_over_years.png', dpi=300, bbox_inches='tight')
plt.show()
  </code>
</pre>

Now, let's have a look at JJAS temperature anomaly over this period.

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

We see that it changed over time but it is not clear whether it has a consistent increase or decrease or no change.

Now, let's have a look at standard deviation of temperature over Asia to see how it varies in different areas of Asia.
<!-- Toggle Button for Image 5 -->
<button onclick="toggleVisibility('image5', 'code5')" style="...">
    Toggle between image and code
</button>

<!-- Image 1 -->
<img src="std_dev_temp_JJAS.png" id="image5" style="display:block;">

<!-- Code Block for Image 1 (initially hidden) -->
<pre id="code5" style="display:none; background-color: #f7f7f7; ...">
  <code>
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

# Open the dataset
ds = xr.open_mfdataset('APHRO_MA_TAVE_025deg_V1808.*.nc')

# Filter the dataset to include only June, July, August, and September
ds_jjas = ds['tave'].sel(time=ds['tave']['time'].dt.month.isin([6, 7, 8, 9]))

# Calculate the standard deviation of 'tave' for JJA across the time dimension
tave_std_jjas = ds_jjas.std(dim='time')

# Compute the standard deviation
tave_std_jjas = tave_std_jjas.compute()

# Define the figure and axis
fig, ax = plt.subplots(figsize=(11, 8.5), subplot_kw={'projection': ccrs.PlateCarree()})
fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05, right=0.95, wspace=0.1, hspace=0.5)

# Dynamically define levels based on the data
min_val = tave_std_jjas.min()
max_val = tave_std_jjas.max()
levels = np.linspace(min_val, max_val, 20)  # Adjust 20 to the desired number of steps

# Make a filled contour plot
cs = ax.contourf(tave_std_jjas['lon'], tave_std_jjas['lat'], tave_std_jjas, 
                 levels=levels, transform=ccrs.PlateCarree(), 
                 cmap='YlOrRd', extend='both')

# Add coastlines and borders
ax.coastlines()
ax.add_feature(cfeature.BORDERS, edgecolor='black')

# Add gridlines and format them
def format_lon(value, tick_number):
    return f"{value:.1f}°E" if value >= 0 else f"{abs(value):.1f}°W"

def format_lat(value, tick_number):
    return f"{value:.1f}°N" if value >= 0 else f"{abs(value):.1f}°S"

ax.gridlines()
ax.set_xticks(np.arange(np.floor(min(ds_jjas['lon'])), np.ceil(max(ds_jjas['lon'])), 15), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(np.floor(min(ds_jjas['lat'])), np.ceil(max(ds_jjas['lat'])), 15), crs=ccrs.PlateCarree())
ax.xaxis.set_major_formatter(plt.FuncFormatter(format_lon))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_lat))

# Add colorbar with increased size
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.04])  # Adjusted y position of colorbar
cbar = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='Standard Deviation (degree Celsius)')
cbar.ax.tick_params(labelsize=10)

# Add title to the map
ax.set_title('Standard Deviation of Temperature for JJAS', fontsize=14, pad=20)

# Save and show the plot
plt.savefig('std_dev_temp_JJAS.png', dpi=300, bbox_inches='tight')
plt.show()
  </code>
</pre>

We can see that standard deviation of temperature varies over Asia. Although it seems that the standard deviation increases as we south.

#### Climatology of temperature in Asia

Now, let's have a look at the climatology of temperature in Asia.

<!-- Toggle Button for Image 1 -->
<button onclick="toggleVisibility('image6', 'code6')" style="...">
    Toggle between image and code
</button>

<!-- Image 6 -->
<img src="climatological_map.png" id="image6" style="display:block;">

<!-- Code Block for Image 6 (initially hidden) -->
<pre id="code6" style="display:none; background-color: #f7f7f7; ...">
  <code>
    import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def extract_monthly_data(filename):
    ds = xr.open_dataset(filename)
    monthly_data = ds['tave'].groupby('time.month').mean('time')
    return monthly_data

def plot_monthly_data(climatology, month, ax, vmin, vmax, levels):
    lons, lats = np.meshgrid(climatology['lon'], climatology['lat'])
    cs = ax.contourf(lons, lats, climatology.sel(month=month), levels=levels, cmap='RdBu_r', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.set_title(month_name(month), fontsize=26)

    # Set longitude labels with larger font size
    lon_formatter = plt.FuncFormatter(lambda x, _: f"{int(x):d}°E" if x >= 0 else f"{int(-x):d}°W")
    ax.set_xticks(np.arange(np.floor(np.min(climatology['lon'])), np.ceil(np.max(climatology['lon'])), 10), crs=ccrs.PlateCarree())
    ax.set_xticklabels([lon_formatter(tick, None) for tick in np.arange(np.floor(np.min(climatology['lon'])), np.ceil(np.max(climatology['lon'])), 10)], fontsize=18)

    # Set latitude labels with larger font size
    lat_formatter = plt.FuncFormatter(lambda x, _: f"{int(x):d}°N" if x >= 0 else f"{int(-x):d}°S")
    ax.set_yticks(np.arange(np.floor(np.min(climatology['lat'])), np.ceil(np.max(climatology['lat'])), 10), crs=ccrs.PlateCarree())
    ax.set_yticklabels([lat_formatter(tick, None) for tick in np.arange(np.floor(np.min(climatology['lat'])), np.ceil(np.max(climatology['lat'])), 10)], fontsize=18)

    return cs

def month_name(month_num):
    import calendar
    return calendar.month_name[month_num]

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(30, 26), subplot_kw={'projection': ccrs.PlateCarree()})

all_data = []
for year in range(1961, 2011):
    filename = f"APHRO_MA_TAVE_025deg_V1808.{year}.nc.nc"
    all_data.append(extract_monthly_data(filename))

combined_data = xr.concat(all_data, dim='year')
climatology = combined_data.mean(dim='year')

vmax = np.abs(climatology.max())
vmin = -vmax

levels = np.linspace(vmin, vmax, 50)

for month, ax in zip(range(1, 13), axes.ravel()):
    cs = plot_monthly_data(climatology, month, ax, vmin, vmax, levels)

cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])
cb = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
cb.ax.tick_params(labelsize=20)
cb.set_label('Temperature (Celsius)', size=22)

fig.suptitle("Climatological Map of South Asia (1961-2010)", fontsize=32, y=0.92)

plt.subplots_adjust(top=0.88, hspace=0.05, wspace=0.1)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('climatological_map.png', dpi=300, bbox_inches='tight')

plt.show()
  </code>
</pre>








<!-- Toggle Button -->
<button onclick="toggleVideoAndCode('video1', 'codeForVideo1')">Toggle between video and code</button>

<!-- Video -->
<video id="video1" width="820" height="640" controls style="display:block;">
  <source src="IOD_Animation_2_Smaller_Size.mp4" type="video/mp4">
  Evolution of IOD Phases
</video>



<!-- Code Block (initially hidden) -->
<pre id="codeForVideo1" style="display:none;">
  <code>
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation

# Make sure that the y2 values in fill_between calls don't contain NaNs. For positive_IOD and negative_IOD, replace NaNs with the respective threshold values.
positive_IOD_filled = positive_IOD.fillna(0.4)  # Fill NaNs with the lower bound for positives
negative_IOD_filled = negative_IOD.fillna(-0.4)  # Fill NaNs with the upper bound for negatives

# Initialize the figure and axis with a specific size
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize plot elements
line, = ax.plot([], [], 'k', linewidth=0.5)

# Update function for the animation
def update(frame):
    current_time = mdates.date2num(dmi['time'][:frame])  # Convert datetime to matplotlib format
    current_sst = dmi['sst'][:frame]

    # Update line plot
    line.set_data(current_time, current_sst)

    # Remove previous fills if they exist
    for coll in [coll for coll in ax.collections]:
        coll.remove()

    # Add new fill areas
    ax.fill_between(current_time, current_sst, where=(current_sst < 0.4) & (current_sst > -0.4), color='lightgreen', label='Neutral IOD')
    ax.fill_between(current_time, 0.4, positive_IOD_filled[:frame], where=positive_IOD_filled[:frame] >= 0.4, color='red', label='Positive IOD', interpolate=True)
    ax.fill_between(current_time, -0.4, negative_IOD_filled[:frame], where=negative_IOD_filled[:frame] <= -0.4, color='blue', label='Negative IOD', interpolate=True)

    return line,

# Set the title, labels, and legend
ax.set_title("Indian Ocean Dipole (IOD) over Time", fontsize=14, weight='bold')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('SST Anomaly', fontsize=12)
ax.legend(loc='best', frameon=False)

# Draw the zero line and the thresholds
ax.axhline(0, color='black', linewidth=0.5)
ax.axhline(1, color='black', linewidth=0.5, linestyle='dotted')
ax.axhline(-1, color='black', linewidth=0.5, linestyle='dotted')

# Improve the x-axis labels with date formatting
ax.xaxis.set_major_locator(mdates.YearLocator(5))  # Major ticks every 5 years
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.YearLocator(1))  # Minor ticks every year

# Add gridlines
ax.grid(which='major', linestyle='-', linewidth='0.5', color='grey')
ax.grid(which='minor', linestyle=':', linewidth='0.5', color='grey')

# Set the layout to be tight to optimize space usage
plt.tight_layout()

# Create the legend manually and place it at the top left
legend_elements = [plt.Line2D([0], [0], color='red', lw=4, label='Positive IOD'),
                   plt.Line2D([0], [0], color='lightgreen', lw=4, label='Neutral IOD'),
                   plt.Line2D([0], [0], color='blue', lw=4, label='Negative IOD')]
ax.legend(handles=legend_elements, loc='upper left')

# Create the animation
ani = FuncAnimation(fig, update, frames=len(dmi['time']), blit=True)

# Add text annotations
data_source_text = "Data Source: NOAA OI SST V2 High Resolution Dataset"
prepared_by_text1 = "Prepared by:"
prepared_by_text2 = "Shammunul Islam"

# Position the 'Data Source' text at the bottom
# ax.text(0.5, 0.01, data_source_text, ha='center', va='bottom', transform=fig.transFigure, fontsize=8)

# Position the 'Prepared by' text, with bold and italic for the name
# ax.text(0.5, 0.05, prepared_by_text, ha='center', va='bottom', transform=fig.transFigure, fontsize=8, style='italic')
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.60, 0.14, prepared_by_text1, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top')
ax.text(0.60, 0.11, prepared_by_text2, transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', style='italic', weight = 'bold')
ax.text(0.60, 0.08, data_source_text, transform=ax.transAxes, fontsize=8, verticalalignment='top')

# Save the animation
ani.save('IOD_timeseries_animation.gif', writer='pillow', fps=20, dpi=300)
plt.legend()
# Show the plot
plt.show()
  </code>
</pre>

### Results 
What does your analysis show that is scientifically interesting? What have you discovered?  

### Summary 
Provide short summary of what you learned from your analysis of your data (both scientific and technical), what you would do next to advance this analysis, and any challenges or issues you encountered/overcame.






























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
