import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# 设置全局样式
plt.rcParams.update({'font.size': 15}) 

os.system("rm formation_energy.txt normalized_distribution.txt plot.png ")
# 读取数据
with open("results_4_eform_m3gnet.csv") as f:
    data = [float(line.strip().split(',')[-1]) for line in f if line.strip().split(',')[-1]!="eform" ]
#print(data)

#cutoff = min(data)
#intvl = abs(cutoff/50.0)
#hist_3d, bins_3d = np.histogram(data, bins=np.arange(cutoff+intvl,0, intvl), density=True)
#percent_y = hist_3d#arr2pctarr(hist_3d)
#plt.bar(bins_3d[:-1],percent_y,width=intvl)

    
# 生成直方图    
props, bins, patches = plt.hist(data, bins=50, density=True, 
                                 edgecolor='black', facecolor='pink',linewidth=0.2, alpha=0.5)

#print(len(props))
#print(len(bins))

outputs = [[str(a), str(b)] for a, b in zip(bins[:-1],props)]
with open('formation_energy.txt','w') as f:
    for op in outputs:
        f.write('\t'.join(op))
        f.write('\n')
    

# 生成渐变颜色并填充  
#cmap = LinearSegmentedColormap.from_list('pink', ['red', 'green', 'blue'], 128)
#for rect in plt.gca().patches:
#    rect.set_color('pink')


#########################
# 绘制正态分布曲线

mean = np.mean(data)
std = np.std(data)

x = np.linspace(mean - 5*std, mean + 5*std, 200)
y = np.exp(-((x - mean)**2)/(2*std**2)) / (std * np.sqrt(2*np.pi))

plt.plot(x, y, linewidth=0.2,linestyle='--',c='b')

with open('normalized_distribution.txt','w') as f:
    for op in zip(x, y):
        f.write('\t'.join([str(num) for num in op]))
        f.write('\n')
    

# 设置坐标轴范围及记号  
plt.xlim(-5,0)
plt.locator_params(axis='x', nbins=5) 
plt.locator_params(axis='y', nbins=5)

# 添加标签          
plt.xlabel('Formation energy per atom (eV)')
plt.ylabel('%Materials')  
#plt.title('Formation energy distribution')

plt.savefig('plot.png') 
plt.show()
os.system("rm formation_energy.txt normalized_distribution.txt  ")
