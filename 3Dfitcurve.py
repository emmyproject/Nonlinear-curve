from termcolor import colored
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits import mplot3d
import matplotlib.font_manager
import matplotlib.colors as mcolors
from astropy.wcs import WCS
#------------------------------------------------------------------------------
# Manual colour map
#------------------------------------------------------------------------------
def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
#-------------------------------------------------------------------------
#format plot
#-------------------------------------------------------------------------
plt.rc('axes', linewidth=1.4)
for tick in plt.gca().xaxis.get_minor_ticks():
    tick.tick1line.set_markeredgewidth(1)
    tick.tick1line.set_markersize(15)
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
wcs = WCS()
plt.tight_layout()

#--------------------------------------------------------------------------
#Import data
#--------------------------------------------------------------------------
data='init.dat'
x,y,z,err_z=np.loadtxt(data,unpack=True,usecols=[0,1,2,3],skiprows=2)
data_3d=np.array([x,y,z])
x_f = np.linspace(0., .5, 100)
y_f = np.linspace(0., 1. , 100)
x_mesh, y_mesh = np.meshgrid(x_f, y_f)
data_mesh=np.array([x_mesh, y_mesh])
nfitparam=22   #Number of fitting parameters

#--------------------------------------------------------------------------
#Defining the fitting function for the imported data 
#--------------------------------------------------------------------------
p0=np.ones(nfitparam)
def fit_crv(x, *a):
    return (a[0] + x[1]*(a[1]+a[2]*x[1])+
            (x[0]**.5)*(a[3]+x[1]*(a[4]+a[5]*x[1]))+
            (x[0]**1.0)*(a[6]+x[1]*(a[7]+a[8]*x[1]))+
            (x[0]**1.5)*(a[9]+x[1]*(a[10]+a[11]*x[1]))+
            (x[0]**2)*(a[12]+x[1]*(a[13]+a[14]*x[1]))+
            (x[0]**2.5)*(a[15]+x[1]*a[16]))\
            /(1.+x[1]*(a[17]+x[1]*(a[18]+a[19]*x[1]+a[20]*x[1]**2))+
              (x[0]**.5)*a[21]*x[1])
#--------------------------------------------------------------------------
#Fit the nonlinear curve to the data
#--------------------------------------------------------------------------
fitParams, fitCovariances = curve_fit(fit_crv,data_3d,z,p0,sigma=z,
                                      method='trf')
#----------------------------------------------------------------------------
#Calculates the histigram of residuals
#----------------------------------------------------------------------------
hist_res=((data_3d[2,:]-fit_crv(data_3d,*fitParams)))*100/data_3d[2,:]

#---------------------------------------------------------------------------
#the value of curve for any given point 
#---------------------------------------------------------------------------
def curve(val):
    return fit_crv(val,*fitParams)

print(" ")
print colored("Fitting parameters :","red")
print colored(fitParams,"red")
print (" ")
#----------------------------------------------------------------------------
#Plot data
#----------------------------------------------------------------------------
#Plot histogram of residuals and point data and fitting curve
#---------------------------------------------------------------------------
ax1 = plt.subplot(1,1,1, projection=wcs)
plt.hist(hist_res, color = 'blue', edgecolor = 'black',bins = int(1./.05))
plt.xlabel('Relative error (%)', size = 18)
plt.ylabel('Number', size= 18)
ax1.tick_params(axis='both', which='both', direction='in')
plt.xticks(np.arange(-5.,5.,1.))
plt.savefig('hist.eps',format='eps',dpi=300, bbox_inches = "tight")
#---------------------------------------------------------------------------
#3D plot 
#---------------------------------------------------------------------------
fig=plt.figure(figsize=(8,8))
ax2=plt.axes(projection='3d')
ax2.scatter(x,y,z,c='r',marker='o')
for i in np.arange(0, len(z)):
    ax2.plot([x[i],x[i]],[y[i],y[i]],[(z[i]-err_z[i]),(z[i]+err_z[i])],
             c='r', marker="_")
ax2.plot_surface(x_mesh, y_mesh,curve(data_mesh),rstride=1, cstride=1,
                color='m')
plt.rcParams['font.family'] = 'serif'
ax2.set_xlabel(r'$x$',fontsize=18,labelpad=12)
ax2.set_xlim(0,.5)
ax2.set_ylabel(r'$y$',fontsize=18,labelpad=12)
ax2.set_zlabel(r'$z$',fontsize=18,labelpad=12,\
              rotation=0)
plt.tight_layout()
plt.figure("fig")
#-------------------------------------------------------------------------
#plot contour
#-------------------------------------------------------------------------
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap(
    [c('purple'), c('violet'), 0.2, c('violet'), c('red'), 0.8, c('red')])
N = 1000
array_dg = np.random.uniform(0, 10, size=(N, 2))
colors = np.random.uniform(-2., 2., size=(N,))
levels=[.2,.25,.3,.35,.4,.45,.5,.55,.6]
ax3 = plt.subplot(1,1,1)
CS = plt.contour(x_mesh,y_mesh,curve(data_mesh),levels,
                 colors=('k',),linestyles=('-',),linewidths=(1,),extend='both',\
                 vmin=0,vmax=5)
CSF=plt.contourf(x_mesh,y_mesh,curve(data_mesh),60,c=colors,cmap=rvb)

plt.clabel(CS,inline=1, fmt = '%2.2f', colors = 'k', fontsize=8)#,manual=loc)
cbr=plt.colorbar(CSF,shrink=.8, extend='both')
plt.xticks(np.arange(0,1.1,.2))
ax3.tick_params(axis='both', which='both', direction='in')
ax3.set_xlabel(r'$x$',fontsize=18)
ax3.set_xlim(0,.5)
ax3.set_ylabel(r'$y$',fontsize=18)
cbr.set_label(r'$z$',fontsize=14,rotation=0,labelpad=-31, y=1.11)
plt.savefig('curve.eps',format='eps',dpi=300, bbox_inches = "tight")
plt.show()



