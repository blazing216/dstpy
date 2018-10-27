#!/usr/bin/env python
#-*-coding:utf-8-*-

import os
import sys
#from subprocess import call, check_output, Popen, PIPE
#sys.path.append('/home/xuyh/jxarray/inversion/multi_scale_inversion/scripts')
#from dst_toolkits import readMOD, readDSurfTomoIn
import numpy as np
#import scipy.interpolate as sinterp
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
#from obspy.geodetics import gps2dist_azimuth

########################################
#      READ DSURFTOMO.IN
########################################

def readDSurfTomoIn(DSurfTomoIn):
    with open(DSurfTomoIn, 'rU') as dsf:
        lines = dsf.readlines()
        nx, ny, nz = lines[4].strip().split()[:3]
        nx, ny, nz = int(nx), int(ny), int(nz)
        max_lat, min_lon = lines[5].strip().split()[:2]
        max_lat, min_lon = float(max_lat), float(min_lon)
        delta_lat, delta_lon = lines[6].strip().split()[:2]
        delta_lat, delta_lon = float(delta_lat), float(delta_lon)

    min_lat = max_lat - (nx-3)*delta_lat
    max_lon = min_lon + (ny-3) * delta_lon

    return nx, ny, nz, min_lat, max_lat, min_lon, max_lon

#########################################
##      READ DSURFTOMO.INMEASURE.dat
#########################################
#
def readDSurfTomoResult(DSurfTomoResult, nx, ny, nz):
    vs3d = np.loadtxt(DSurfTomoResult, dtype=[('lon', np.float64),
                                              ('lat', np.float64),
                                              ('dep', np.float64),
                                              ('vs',  np.float64)])
    vs = vs3d['vs']
    lon = vs3d['lon']
    lat = vs3d['lat']
    dep = vs3d['dep']

    lon.shape = nz-1, ny-2, nx-2
    lat.shape = nz-1, ny-2, nx-2
    dep.shape = nz-1, ny-2, nx-2
    vs.shape  = nz-1, ny-2, nx-2

    # revert lat to change latitude from descending to ascending
    # revert vs accordingly
    # revert lon, dep does not change lon, dep acctually, so we skip it
    lat = lat[:,:,::-1]
    vs  =  vs[:,:,::-1]

    extent = [lon[0,0,0], lon[0,-1,0], lat[0,0,0], lat[0,0,-1]]
    #return vs, extent, dep[:,0,0], lon[0,:,0], lat[0,0,:]
    return vs, dep[:,0,0]

def readMOD(MODfile, nx, ny, nz):
    vs = np.zeros((nz, ny, nx))
    with open(MODfile, 'rU') as mf:
        lines = mf.readlines()

    dep = [float(d) for d in lines[0].strip().split()]
    dep = np.array(dep)

    i = 0
    for k in range(nz):
        for j in range(ny):
            i += 1
            vs[k,j,:] = np.array([float(v) for v in \
                                  lines[i].strip().split()])

    vs = vs[:,:,::-1]

    return vs, dep

def readDWS(MODfile, nx, ny, nz):
    dws = np.zeros((nz-1, ny-2, nx-2))
    with open(MODfile, 'rU') as mf:
        lines = mf.readlines()

    dep = [float(d) for d in lines[0].strip().split()]
    dep = np.array(dep)

    i = 0
    for k in range(nz-1):
        for j in range(ny-2):
            i += 1
            dws[k,j,:] = np.array([float(v) for v in \
                                  lines[i].strip().split()])

    dws = dws[:,:,::-1]

    return dws, dep

def writeDWS(DWSfile, nx, ny, nz, dep, dws):
    dws = dws[:,:,::-1]

    if isinstance(DWSfile, file):
        out = DWSfile
    else:
        out = open(DWSfile, 'w')

    out.write(' '.join(['%.3f' % d for d in dep]) + '\n')
    for k in range(nz-1):
        for j in range(ny-2):
            out.write(' '.join(['%.6f' % d for d in dws[k,j,:]]) + '\n')


def plotVs(vs, dep, dws=None, extent=None, axis_limit=None, interpolation='none',
           cmap=None, vmin=None, vmax=None, plot_colorbar=False,
           axs=None, cb_label=None, plottype='image', levels=None):
    #nx, ny, nz, min_lat, max_lat, min_lon, max_lon = \
    #        readDSurfTomoIn(DSurfTomoIn)
    #d_lat = (max_lat - min_lat) / (nx-3)
    #d_lon = (max_lon - min_lon) / (ny-3)
    #extent = [min_lon-0.5*d_lon, max_lon+0.5*d_lon,
    #          min_lat-0.5*d_lat, max_lat+0.5*d_lat]
    #vs, dep = readMOD(MODfile, nx, ny, nz)
    #vs = vs[:-1,1:-1, 1:-1]

    #vs_init = readMOD(InitMODfile, nx, ny, nz)[0]
    #vs_init = vs_init[:-1,1:-1, 1:-1]

    #print (vs-vs_init).min(), (vs-vs_init).max()


    if isinstance(plottype,str):
        plottype = [plottype]

    if 'contour' in plottype:
        lons = np.linspace(axis_limit[0], axis_limit[1], vs.shape[1])
        lats = np.linspace(axis_limit[2], axis_limit[3], vs.shape[2])

    if axs is None:
        fig, axs = plt.subplots(4, 3, figsize=(10, 10),
                                gridspec_kw=dict(wspace=0.1,hspace=0.3))
    for i in range(4):
        for j in range(3):
            k = j + i*3
            if k >= vs.shape[0]:
                ax = axs[i,j]
                ax.set_visible(False)
                continue
            ax = axs[i,j]


            for pt in plottype:

                if pt == 'image':
                    print 'image', extent, axis_limit
                    im = ax.imshow(vs[k,:,:].T, cmap=cmap, origin='lower',
                                   extent=extent, interpolation=interpolation,
                                   vmin=vmin, vmax=vmax)


                elif pt == 'contour':
                    print 'contour', axis_limit

                    if levels is not None:
                        cs = ax.contour(lons, lats, vs[k,:,:].T, levels=levels, 
                                        colors='k', linewidths=0.5)
                        #ax.clabel(cs, [levels[0], levels[-1]])
                    else:
                        cs = ax.contour(lons, lats, vs[k,:,:].T, colors='k', linewidths=0.5)
                    ax.clabel(cs, cs.levels)
                    #plot_colorbar = False

                else:
                    warnings.warn('Plot type %s not supported. Skip.' %\
                                  plottype)

            ax.text(1,1, '%.1f-%.1f km' % (dep[k], dep[k+1]), ha='right', va='top',
                    transform=ax.transAxes,
                    bbox=dict(pad=3, facecolor='lightgray'))
            ax.set_aspect(1.0)
            #print axis_limit
            if axis_limit is not None:
                ax.axis(axis_limit)

            if plot_colorbar:
                if vmin is not None and vmax is not None:
                    cb = plt.colorbar(im, ax=ax,
                                 #ticks=np.arange(vmin, vmax, 0.3),
                                 #vmin=vmin, vmax=vmax,
                                 label=cb_label)
                else:
                    cb = plt.colorbar(im, ax=ax,
                                 #ticks=np.arange(vmin, vmax, 0.3),
                                 label=cb_label)
                if j != 2:
                    cb.ax.set_visible(False)
    return axs

def plotMOD(DSurfTomoIn, MODfile, interpolation='none', vmin=0.5,
            vmax=4.0, axs=None, cmap=None, plot_colorbar=False):

    #plt.style.use('gmt')
    #mpl.rcParams['axes.labelsize'] = 14
    #mpl.rcParams['xtick.labelsize'] = 12
    #mpl.rcParams['ytick.labelsize'] = 12

    for dstf, modf in zip(DSurfTomoIn, MODfile):
        nx, ny, nz, min_lat, max_lat, min_lon, max_lon = \
                readDSurfTomoIn(dstf)
        d_lat = (max_lat - min_lat) / (nx-3)
        d_lon = (max_lon - min_lon) / (ny-3)
        extent = [min_lon-0.5*d_lon, max_lon+0.5*d_lon,
                  min_lat-0.5*d_lat, max_lat+0.5*d_lat]
        axis_limit = [min_lon, max_lon, min_lat, max_lat]
        vs, dep = readMOD(modf, nx, ny, nz)
        vs = vs[:-1,1:-1, 1:-1]

        #vs_init = readMOD(InitMODfile, nx, ny, nz)[0]
        #vs_init = vs_init[:-1,1:-1, 1:-1]

        print vs.min(), vs.max()

        #fig, axs = plt.subplots(4, 3, figsize=(10, 10),
        #                        gridspec_kw=dict(wspace=0.1,hspace=0.3))
        plotVs(vs, dep, dws=None, extent=extent, axis_limit=axis_limit,
               interpolation=interpolation,
               cmap=cmap, vmin=vmin, vmax=vmax, plot_colorbar=True,
              axs=axs, cb_label='Vs (km/s)', plottype='image')

    #for i in range(4):
    #    for j in range(3):
    #        k = j + i*3
    #        if k >= vs.shape[0]:
    #            ax = axs[i,j]
    #            ax.set_visible(False)
    #            continue
    #        ax = axs[i,j]

    #        im = ax.imshow(vs[k,:,:].T - vs_init[k,:,:].T, cmap='seismic_r', origin='lower',
    #                       extent=extent, interpolation='none',
    #                       vmin=vmin, vmax=vmax)
    #        ax.text(1,1, '%.1f-%.1f km' % (dep[k], dep[k+1]), ha='right', va='top',
    #                transform=ax.transAxes,
    #                bbox=dict(pad=3, facecolor='lightgray'))
    #        ax.set_aspect(1.0)
    #        ax.set_xlim(min_lon, max_lon)
    #        ax.set_ylim(min_lat, max_lat)
    #        cb = fig.colorbar(im, ax=ax,
    #                     ticks=np.arange(vmin, vmax, 0.3),
    #                     label='Vs residuals (km/s)')
    #        if j != 2:
    #            cb.ax.set_visible(False)
    return axs

def plotMODdiff(DSurfTomoIn, MODfile, InitMODfile, interpolation='none', vmin=0.5,
            vmax=4.0, axs=None, cmap=None):

    #plt.style.use('gmt')
    #mpl.rcParams['axes.labelsize'] = 14
    #mpl.rcParams['xtick.labelsize'] = 12
    #mpl.rcParams['ytick.labelsize'] = 12

    for dstf, modf, initmodf in zip(DSurfTomoIn, MODfile, InitMODfile):
        nx, ny, nz, min_lat, max_lat, min_lon, max_lon = \
                readDSurfTomoIn(dstf)
        d_lat = (max_lat - min_lat) / (nx-3)
        d_lon = (max_lon - min_lon) / (ny-3)
        extent = [min_lon-0.5*d_lon, max_lon+0.5*d_lon,
                  min_lat-0.5*d_lat, max_lat+0.5*d_lat]
        axis_limit = [min_lon, max_lon, min_lat, max_lat]
        vs, dep = readMOD(modf, nx, ny, nz)
        vs = vs[:-1,1:-1, 1:-1]

        vs_init = readMOD(initmodf, nx, ny, nz)[0]
        vs_init = vs_init[:-1,1:-1, 1:-1]

        print (vs-vs_init).min(), (vs-vs_init).max()

        plotVs(vs-vs_init, dep, dws=None, extent=extent, axis_limit=axis_limit,
               interpolation=interpolation,
               cmap=cmap, vmin=vmin, vmax=vmax, plot_colorbar=True,
              axs=axs, cb_label='Vs residual (km/s)', plottype='image')

    #fig, axs = plt.subplots(4, 3, figsize=(10, 10),
    #                        gridspec_kw=dict(wspace=0.1,hspace=0.3))
    #for i in range(4):
    #    for j in range(3):
    #        k = j + i*3
    #        if k >= vs.shape[0]:
    #            ax = axs[i,j]
    #            ax.set_visible(False)
    #            continue
    #        ax = axs[i,j]

    #        im = ax.imshow(vs[k,:,:].T - vs_init[k,:,:].T, cmap='seismic_r', origin='lower',
    #                       extent=extent, interpolation='none',
    #                       vmin=vmin, vmax=vmax)
    #        ax.text(1,1, '%.1f-%.1f km' % (dep[k], dep[k+1]), ha='right', va='top',
    #                transform=ax.transAxes,
    #                bbox=dict(pad=3, facecolor='lightgray'))
    #        ax.set_aspect(1.0)
    #        ax.set_xlim(min_lon, max_lon)
    #        ax.set_ylim(min_lat, max_lat)
    #        cb = fig.colorbar(im, ax=ax,
    #                     ticks=np.arange(vmin, vmax, 0.3),
    #                     label='Vs residuals (km/s)')
    #        if j != 2:
    #            cb.ax.set_visible(False)
    return axs

def plotMODdiff_abs(DSurfTomoIn, MODfile, InitMODfile, interpolation='none', vmin=0.5,
            vmax=4.0, axs=None, cmap=None):

    for dstf, modf, initmodf in zip(DSurfTomoIn, MODfile, InitMODfile):
        nx, ny, nz, min_lat, max_lat, min_lon, max_lon = \
                readDSurfTomoIn(dstf)
        d_lat = (max_lat - min_lat) / (nx-3)
        d_lon = (max_lon - min_lon) / (ny-3)
        extent = [min_lon-0.5*d_lon, max_lon+0.5*d_lon,
                  min_lat-0.5*d_lat, max_lat+0.5*d_lat]
        axis_limit = [min_lon, max_lon, min_lat, max_lat]
        vs, dep = readMOD(modf, nx, ny, nz)
        vs = vs[:-1,1:-1, 1:-1]

        vs_init = readMOD(initmodf, nx, ny, nz)[0]
        vs_init = vs_init[:-1,1:-1, 1:-1]

        print (np.abs(vs-vs_init)).min(), (np.abs(vs-vs_init)).max()

        plotVs(np.abs(vs-vs_init), dep, dws=None, extent=extent, axis_limit=axis_limit,
               interpolation=interpolation,
               cmap=cmap, vmin=vmin, vmax=vmax, plot_colorbar=True,
              axs=axs, cb_label='Vs residual (km/s)', plottype='image')

    return axs

def plotDWS(DSurfTomoIn, DWSfile, interpolation='none', vmin=0.5,
            vmax=4.0, cmap=None, axs=None, plottype='contour',
           plot_colorbar=False, cb_label=None,
           levels=None):

    #plt.style.use('gmt')
    #mpl.rcParams['axes.labelsize'] = 14
    #mpl.rcParams['xtick.labelsize'] = 12
    #mpl.rcParams['ytick.labelsize'] = 12

    for dstf, dwsf in zip(DSurfTomoIn, DWSfile):
        nx, ny, nz, min_lat, max_lat, min_lon, max_lon = \
                readDSurfTomoIn(dstf)
        d_lat = (max_lat - min_lat) / (nx-3)
        d_lon = (max_lon - min_lon) / (ny-3)
        extent = [min_lon-0.5*d_lon, max_lon+0.5*d_lon,
                  min_lat-0.5*d_lat, max_lat+0.5*d_lat]
        axis_limit = [min_lon, max_lon, min_lat, max_lat]
        vs, dep = readDWS(dwsf, nx, ny, nz)
        #vs = vs[:-1,1:-1, 1:-1]

        #vs_init = readMOD(InitMODfile, nx, ny, nz)[0]
        #vs_init = vs_init[:-1,1:-1, 1:-1]

        print vs.min(), vs.max()

        plotVs(vs,dep, dws=None, extent=extent, axis_limit=axis_limit,
               cmap=cmap, interpolation=interpolation, vmin=vmin, vmax=vmax,
               plot_colorbar=plot_colorbar,
              axs=axs,  plottype=plottype, cb_label=cb_label, 
              levels=levels)

    #fig, axs = plt.subplots(4, 3, figsize=(10, 10),
    #                        gridspec_kw=dict(wspace=0.1,hspace=0.3))
    #for i in range(4):
    #    for j in range(3):
    #        k = j + i*3
    #        if k >= vs.shape[0]:
    #            ax = axs[i,j]
    #            ax.set_visible(False)
    #            continue
    #        ax = axs[i,j]


    #        im = ax.imshow(vs[k,:,:].T - vs_init[k,:,:].T, cmap='seismic_r', origin='lower',
    #                       extent=extent, interpolation='none',
    #                       vmin=vmin, vmax=vmax)
    #        ax.text(1,1, '%.1f-%.1f km' % (dep[k], dep[k+1]), ha='right', va='top',
    #                transform=ax.transAxes,
    #                bbox=dict(pad=3, facecolor='lightgray'))
    #        ax.set_aspect(1.0)
    #        ax.set_xlim(min_lon, max_lon)
    #        ax.set_ylim(min_lat, max_lat)
    #        cb = fig.colorbar(im, ax=ax,
    #                     ticks=np.arange(vmin, vmax, 0.3),
    #                     label='Vs residuals (km/s)')
    #        if j != 2:
    #            cb.ax.set_visible(False)
    #return fig
    return axs



#if __name__ == '__main__':
#    usage = '%s DSurfTomoIn InvertedMOD InitMOD vmin vmax' % sys.argv[0]
#    if len(sys.argv) != 6:
#        print usage
#        sys.exit()
#
#    vmin = float(sys.argv[4])
#    vmax = float(sys.argv[5])
#
#    fig = plotMOD(sys.argv[1], sys.argv[2], sys.argv[3], vmin, vmax)
#    fig.savefig('depth_slice.cb.%s.pdf' % sys.argv[2])
#    print 'depth_slice.cb.%s.pdf' % sys.argv[2]

    #nx, ny, nz, min_lat, max_lat, min_lon, max_lon = \
    #        readDSurfTomoIn(sys.argv[1])
    #d_lat = (max_lat - min_lat) / (nx-3)
    #d_lon = (max_lon - min_lon) / (ny-3)
    #extent = [min_lon-0.5*d_lon, max_lon+0.5*d_lon,
    #          min_lat-0.5*d_lat, max_lat+0.5*d_lat]
    #vs, dep = readMOD(sys.argv[2], nx, ny, nz)


    #fig, axs = plt.subplots(4, 3, figsize=(10, 10),
    #                        sharex=True, sharey=True,
    #                        gridspec_kw=dict(wspace=0.3))
    #for i in range(4):
    #    for j in range(3):
    #        k = j + i*3
    #        if k >= vs.shape[0]:
    #            ax = axs[i,j]
    #            ax.set_visible(False)
    #            continue
    #        ax = axs[i,j]

    #        im = ax.imshow(vs[k,:,:].T, cmap='seismic_r', origin='lower',
    #                       extent=extent, interpolation='bicubic',
    #                       vmin=2.0, vmax=4.0)
    #        ax.text(1,1, '%.1f km' % dep[k], ha='right', va='top',
    #                transform=ax.transAxes,
    #                bbox=dict(pad=3, facecolor='lightgray'))
    #        ax.set_aspect(1.0)
    #        ax.set_xlim(min_lon, max_lon)
    #        ax.set_ylim(min_lat, max_lat)
    #        fig.colorbar(im, ax=ax,
    #                     ticks=np.arange(2.0, 4.1, 0.2),
    #                     label='Vs (km/s)')
    #plt.savefig('depth_slice.%s.png' % sys.argv[2])
    #print 'depth_slice.%s.png' % sys.argv[2]


