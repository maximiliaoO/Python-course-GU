# import packages
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read image
# greyscale
# blur

img = cv2.imread('DEimage.tif',0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

#detect circles
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,7,
               param1=290,param2=100,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


### CREATE DATAFRAME WITH IMAGE RECOGNITION
# Detect droplets
# Index droplets, variable i_drop
# Create dataframe df_droplets with i_drop
# Measure droplet diameter, variable dia_drop, add as column
# Detect double-emulsion droplets, variable dble_emul, add as column
# Measure inner droplet diameter for double emulsions, variable dia_inner_drop, add as column
# Detect number of cells in droplet, variable n_cell, add as column

# import example dataframe from csv
df_droplets = pd.read_csv('df_droplets.csv')



### PLOT HISTOGRAM OF NUMBER OF CELLS IN DOUBLE-EMULSIONS

n_cells_histo = df_droplets.hist(column = 'n_cells', bins=3)
plt.title('Histogram number of cells per droplet')
plt.xlabel('# of cells in droplet')
plt.xticks(np.arange(0,2.1,1))
plt.yticks(np.arange(0,18.1,1))

plt.savefig('n_cells_histo.png')

# save data of histogram in CSV table

drops_table = df_droplets['dble_emul'].value_counts()
drops_table = drops_table.to_frame()
drops_table.columns = ['number of occurances']
drops_table.index.name = 'number of cells in droplet'

drops_table.to_csv('droplets_histo_table.csv')



### DATAWRANGLING

# define function for volume

def volume(diameter):
    vol = 4/3*np.pi*((diameter/2)**3)
    return vol

# add column to df_droplets with volume of droplet, variable v_drop = 4/3*pi*dia_drop^3

df_droplets['volume'] = df_droplets['dia_drop'].apply(volume)

#add column to df_droplets with volume of droplet, variable v_inner_drop = 4/3*pi*dia_inner_drop^3
#if dbl_emul > 1 add NaN to column

df_droplets['inner_volume'] = np.where(df_droplets['dble_emul'] > 1 , np.nan, df_droplets['dia_inner_drop'].apply(volume))

#save dataframe to CSV
df_droplets = df_droplets.round(2)
df_droplets.to_csv('dataframe_droplets.csv')



### SUMMARY TABLE

# length of df_droplets is total number of droplets detected, assign as variable n_drops
n_drops = len(df_droplets.index)

# find number of double emulsiondroplets (meaining where dble_emul is 1), assign as variable n_de_drops
n_de_drops = (df_droplets['dble_emul'] == 1).sum()

# calculate percentage of double-emulsion droplets, variable percent_de = n_drops/n_de_drops
percent_de = n_de_drops/n_drops

# calculate mean volume and standard deviation of volume from df_droplets
mean_vol = np.mean(df_droplets['volume'])
std_vol = np.std(df_droplets['volume'])

# calculate mean volume and standard deviation of volume from df_droplets for inner droplet
de_true = df_droplets[df_droplets["dble_emul"] == 1] #create dataframe of droplets that have single double-emulsion

mean_inner_vol = np.mean(de_true['inner_volume'])
std_inner_vol = np.std(de_true['inner_volume'])

# calculate mean and standard deviation for number of cells per double-emulsion droplets
mean_cells = np.mean(de_true['n_cells'])
std_cells = np.std(de_true['n_cells'])

# create df_summary with above calculated values
ls_summary = [n_drops, n_de_drops, percent_de, mean_vol, std_vol, mean_inner_vol, std_inner_vol, mean_cells, std_cells]
df_summary = pd.DataFrame(ls_summary, index=['# droplets', '# DE droplets', '% DE droplets', 'mean volume', 'std volume', 'mean inner volume', 'std inner volume', 'mean cells per DE droplet', 'std cells per DE droplet'])
df_summary = df_summary.round(2)
df_summary.columns = ['value']
df_summary.to_csv('summary.csv')

# https://learnopencv.com/edge-detection-using-opencv/ 







