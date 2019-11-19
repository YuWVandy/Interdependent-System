# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 15:06:57 2019

@author: wany105
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 21:17:47 2019

@author: wany105
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:59:05 2019

@author: wany105
"""
## This file is to:
# set the environment
# Create the map of targeted area using Basemap Package and Assign population count to each square on the map.
## The network and the flow added on it will be simulated based on the population Count of the area.


#Package loading and environment setting
import os
os.environ['PROJ_LIB'] = r'C:\Users\wany105\AppData\Local\Continuum\anaconda3\pkgs\proj4-5.2.0-ha925a31_1\Library\share'

"""
The environment is different among different computers. Please check where the epsg file is in your own computer and change the path in os.environ accordingly
EXAMPLE: In my laptop ---- os.environ['PROJ_LIB'] = r'E:\Anaconda\Library\share'
"""

from mpl_toolkits.basemap import Basemap ##Basemap package is used for creating geography map
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import math
import csv

##Color Setting, different value on the map will have different color
def custom_div_cmap(numcolors, mincol, midlowcol, midcol, midhighcol, maxcol):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(numcolors, colors = [mincol, midlowcol, midcol, midhighcol, maxcol], N = numcolors)

    return cmap
cmap = colors.ListedColormap(['darkblue', 'mediumblue', 'skyblue', 'lightgray', 'khaki', 'yellow', 'orange', 'coral', 'orangered', 'red', 'darkred', 'maroon'])
bounds=[3, 15, 27, 31, 2174, 4318, 6462, 8606, 10750, 12893, 17181, 21469] ##Need to be adjusted to different value
norm = colors.BoundaryNorm(bounds, cmap.N)


# Open the population data file and make sure it is in the same folder with current program file
# The file can be found on the website: https://daac.ornl.gov/ISLSCP_II/guides/global_population_xdeg.html (Population density, population and area of each area)
filename = 'people_count_1995.asc' #Population map, not density!
with open(filename) as f:
    reader = csv.reader(f)
    line_count = 0
    Title = []
    #Set up the grid and split the whole map into small squares: Latitude, 90*2*4; longitude: 180*2*4.
    #PD: Population Count of each small square area.
    PD = np.zeros([720, 1440]) 
    for row in reader:
        if(line_count <= 5):
            Title.append(row)
            line_count += 1
        else:
            Temp = row[0].split()
            temp2 = 0
            for temp in Temp:
                if((float(temp) == -88) or (float(temp) == -99)):
                    PD[719 - (line_count - 6)][temp2] = 0
                else:
                    PD[719 - (line_count - 6)][temp2] = temp
                temp2 += 1
            line_count += 1



#------------------------------------------------------------------------------Basic Variable Input
##Coordinates of the whole Earth
"""
llon = -180
llat = -90
rlon = 180
rlat = 90
Type = 'whole'
"""
##Coordinates of the boundary of the targeted area
llon, rlon = -97.5, -82.5
llat, rlat = 27.5, 42.5
Type = 'local' #Global or Local

##Coordinates of the boundary of the earthquake can be
Disrupllon, Disruprlon = -100, -80
Disrupllat, Disruprlat = 25, 45


#eq_map contains state plane coordinates of every square in the map we specified          
if(Type == 'local'):
    eq_map = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=rlon, urcrnrlat=rlat)
elif(Type == 'whole'):
    eq_map = Basemap(resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon=llon, llcrnrlat=llat, urcrnrlon=rlon, urcrnrlat=rlat)

#eq_map contains state plane coordinates of every square in the map we specified          
if(Type == 'local'):
    Disrupeq_map = Basemap(projection = 'merc', resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon=Disrupllon, llcrnrlat=Disrupllat, urcrnrlon=Disruprlon, urcrnrlat=Disruprlat)
elif(Type == 'whole'):
    Disrupeq_map = Basemap(resolution = 'l', area_thresh = 1000.0, lat_0=0, lon_0=0, llcrnrlon=Disrupllon, llcrnrlat=Disrupllat, urcrnrlon=Disruprlon, urcrnrlat=Disruprlat)



#------------------------------------------------------------------------------Population Import
#PDsub: Population Count of small square area in targeted area we specified before
d_lat = 0.25
d_lon = 0.25
PDsub = np.zeros([int((rlat-llat)/0.25), int((rlon - llon)/0.25)])
PDsub = PD[int(math.floor((llat-(-90))/d_lat)):int(math.floor((rlat-(-90))/d_lat)), int(math.floor((llon-(-180))/d_lon)):int(math.floor((rlon-(-180))/d_lon))]

DisrupPDsub = np.zeros([int((Disruprlat-Disrupllat)/0.25), int((Disruprlon - Disrupllon)/0.25)])
DisrupPDsub = PD[int(math.floor((Disrupllat-(-90))/d_lat)):int(math.floor((Disruprlat-(-90))/d_lat)), int(math.floor((Disrupllon-(-180))/d_lon)):int(math.floor((Disruprlon-(-180))/d_lon))]
lon = np.arange(llon, rlon + d_lon, d_lon) #Set up the grid on our specified area and calculate the lat and lon of each square
lat = np.arange(llat, rlat + d_lat, d_lat)

Disruplon = np.arange(Disrupllon, Disruprlon + d_lon, d_lon) #Set up the grid on our specified area and calculate the lat and lon of each square
Disruplat = np.arange(Disrupllat, Disruprlat + d_lat, d_lat)

#Transfer lon and Lat coordinates to state plane coordinates
for i in range(len(Disruplon)):
    Disruplon[i], temp = Disrupeq_map(Disruplon[i], 0)

for i in range(len(Disruplat)):
    temp, Disruplat[i] = Disrupeq_map(0, Disruplat[i])


#Transfer lon and Lat coordinates to state plane coordinates
for i in range(len(lon)):
    lon[i] = Disruplon[i + int((llon - Disrupllon)/0.25)]

for i in range(len(lat)):
    lat[i] = Disruplat[i + int((llat - Disrupllat)/0.25)]



