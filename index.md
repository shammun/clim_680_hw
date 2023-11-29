# Impact of Indian Ocean Dipole on Temperature in Asia 

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

#### 

### Code description 
Description of each of your analyses along with a link to your notebook for each analysis. This can be in a bullet or table of contents type format.  



<video width="320" height="240" controls>
  <source src="IOD_Animation_2_Smaller_Size.mp4" type="video/mp4">
  Evolution of IOD Phases
</video>


















<!-- Toggle Button -->
<button onclick="toggleVisibility('image1', 'code1')">Toggle between image and code</button>

<!-- Image -->
<img src="1.png" id="image1" style="display:block;">

<!-- Code Block (initially hidden) -->
<pre id="code1" style="display:none;">
  <code>
    // Your code here
    import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import cartopy.feature as cfeature

# Assuming ds_mean_1961 and ds_1961 are your data sources

# Make the figure larger
fig = plt.figure(figsize=(11, 8.5))

# Set the axes using the specified map projection
# Adjust the position of the map to leave space for the colorbar
ax = plt.axes([0.05, 0.2, 0.9, 0.7], projection=ccrs.PlateCarree())

# Add cyclic point to data
data = ds_mean_1961['tave']
data, lons = add_cyclic_point(data, coord=ds_1961['lon'])

# Define levels for finer intervals, ignoring NaNs
level_interval = 2  # Change this value as needed
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
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03]) # Adjust these values as needed
cbar = plt.colorbar(cs, cax=cbar_ax, orientation='horizontal', label='Surface Air Temperature (Celsius)')

# Save and show the plot
plt.savefig('air_temp_1961.png', dpi=300, bbox_inches='tight')
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


