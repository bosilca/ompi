#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import sys

print("\n \n Processing first file \n")

filename1 = sys.argv[1]
f1 = open(filename1,"r")
lines1 = f1.readlines()
f1.close()

noMin1 = {}
sortednomin1 = {}

#Select lines with key
for line in lines1:
   if "time" in line:
        vals = line.split( )
        noMin1.setdefault((int)(vals[2]), []).append((float)(vals[4]))

sortednomin1=sorted(noMin1.items())
num_of_elements1 = [x[0] for x in sortednomin1]
Reduce_latency1 = [x[1] for x in sortednomin1]
print(Reduce_latency1)

print("\n \n Processing second file \n")


filename2 = sys.argv[2]
f2 = open(filename2,"r")
lines2 = f2.readlines()
f2.close()

noMin2 = {}
sortednomin2 = {}


#Select lines with key
for line in lines2:
   if "time" in line:
        vals = line.split( )
        noMin2.setdefault((int)(vals[2]), []).append((float)(vals[4]))

sortednomin2=sorted(noMin2.items())
num_of_elements2 = [x[0] for x in sortednomin2]
Reduce_latency2 = [x[1] for x in sortednomin2]
print(Reduce_latency2)

Msg=[1,2,3,4,5,6,7,8,9]

Msg_size=['1k', '4k', '16k', '64k', '256k', '1M', '4M', '16M', '32M']

meanlineprops = dict(linewidth=2)
fig, (ax1) = plt.subplots(1, figsize=(6, 4))

ax1.grid(True,linestyle='--')


bp1 =ax1.boxplot(Reduce_latency1, notch=1, sym='+', vert=1, whis=1.5,patch_artist=True)
bp2 =ax1.boxplot(Reduce_latency2, notch=0, sym='+', vert=1, whis=1.5,patch_artist=True)
#bp3 =ax1.boxplot(cpy, notch=0, sym='+', vert=1, whis=1.5,patch_artist=True)
#ax1.plot(Msg, cpy, linestyle='-.', lw=2.5, marker='o',markersize='8', color='b', label="0.1s")

colors = ['red','red','red','red','red','red','red','red','red']
for patch, color in zip(bp2['boxes'], colors):
    patch.set_facecolor(color)

colors = ['purple','purple','purple','purple','purple','purple','purple','purple','purple']
for patch, color in zip(bp1['boxes'], colors):
    patch.set_facecolor(color)

"""
colors = ['green','green','green','green','green','green']
for patch, color in zip(bp3['boxes'], colors):
    patch.set_facecolor(color)
"""
fig.suptitle('MPI_Op: mul; MPI_type: uint8 \n(Configured with optimized, Flushed cache)', x=0.45, y=1.01, va='center',fontsize=15)
fig.text(-0.03, 0.5, 'Reduce time (s)', va='center', rotation='vertical',fontsize=12)
plt.xlabel('Buffer size (Bytes)',fontsize=12)

ax1.set_xticks(Msg)
ax1.set_xticklabels(Msg_size, minor=False)


ax1.legend([bp2["boxes"][0], bp1["boxes"][0]], ['NO_SVE','SVE'], loc='upper left',fontsize=12)#, bbox_to_anchor=(1, 0.5),fontsize=15)

plt.yscale('log')

plt.savefig(filename1+"_sum.pdf",dpi=300,bbox_inches='tight')
