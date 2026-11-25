"""Export exact environmental inputs behind Figure 3C, Figure 4 and SI circulation summaries.

Run after staging provider NetCDF/rainfall/runoff files as described in data/README.md.
These outputs are Eulerian environmental context, not larval-transport estimates.
"""
from pathlib import Path
import importlib.util
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR))
from analysis_inputs import annual_discharge,ipo_phase_years,reef_coordinates,ecological_observations
from config import DATA_DIR,RESULTS_DIR

def load_script(filename):
    spec=importlib.util.spec_from_file_location(filename.replace('.','_'),SCRIPT_DIR/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    out=RESULTS_DIR/'source_data';out.mkdir(parents=True,exist_ok=True)
    neg,pos,ipo=ipo_phase_years()
    mech=load_script('06_mechanism_fig6.py')
    rain=mech.process_rainfall(neg,pos)
    chl,chl_ts=mech.process_chlorophyll(neg,pos)
    with xr.open_dataset(DATA_DIR/'climate/GBR_currents_monthly_1993_2025.nc') as ds:
        monthly=ds.vo
        counts=pd.Series(monthly.time.dt.year.values).value_counts()
        complete=monthly.sel(time=monthly.time.dt.year.isin(counts.index[counts.eq(12)]))
        annual=complete.resample(time='YE').mean(dim='time',skipna=False).load().squeeze('depth',drop=True)
    ng=annual.sel(time=annual.time.dt.year.isin(neg));pg=annual.sel(time=annual.time.dt.year.isin(pos))
    vn=ng.mean('time');vp=pg.mean('time')
    np.savez_compressed(out/'environment_maps.npz',
        rain_lat=rain.lat.values,rain_lon=rain.lon.values,rain_neg_minus_pos=rain.values,
        chl_lat=chl.latitude.values,chl_lon=chl.longitude.values,chl_neg_minus_pos=chl.values,
        current_lat=vn.latitude.values,current_lon=vn.longitude.values,
        current_negative=vn.values,current_positive=vp.values,current_neg_minus_pos=(vn-vp).values)
    series=pd.concat([chl_ts.rename('chlorophyll_mg_m3'),ipo.rename('unfiltered_tpi')],axis=1).dropna()
    series.index.name='year';series.to_csv(out/'chlorophyll_tpi_annual.csv')
    rivers=pd.concat([annual_discharge(n).rename(n) for n in ['Pascoe','Burdekin','Fitzroy']],axis=1)
    rivers.index.name='year';rivers.to_csv(out/'river_discharge_annual_m3_s.csv')
    cots,_=ecological_observations()
    coords=reef_coordinates().merge(cots[['reef_name','nrm_region']].drop_duplicates(),on='reef_name',validate='one_to_one')
    rlat=xr.DataArray(coords.lat.to_numpy(),dims='reef');rlon=xr.DataArray(coords.lon.to_numpy(),dims='reef')
    coords['positive_vo_m_s']=vp.sel(latitude=rlat,longitude=rlon,method='nearest').values
    coords['negative_vo_m_s']=vn.sel(latitude=rlat,longitude=rlon,method='nearest').values
    coords['difference_vo_m_s']=coords.negative_vo_m_s-coords.positive_vo_m_s
    coords.to_csv(out/'reef_current_phase_composites.csv',index=False)
    summary=coords.groupby('nrm_region').agg(n=('reef_name','size'),positive_vo_m_s=('positive_vo_m_s','mean'),negative_vo_m_s=('negative_vo_m_s','mean'))
    southward=summary.positive_vo_m_s.lt(0)&summary.negative_vo_m_s.lt(0)
    summary['southward_weakening_percent']=np.where(southward,100*(summary.positive_vo_m_s-summary.negative_vo_m_s)/summary.positive_vo_m_s,np.nan)
    summary.to_csv(out/'regional_current_phase_composites.csv')
    meta={'current_complete_years':[int(y) for y in annual.time.dt.year.values],
          'current_negative_years':[int(y) for y in ng.time.dt.year.values],
          'current_positive_years':[int(y) for y in pg.time.dt.year.values],
          'positive_northward_difference_reefs':int(coords.difference_vo_m_s.gt(0).sum()),
          'sampled_reefs':len(coords),'current_depth_m':0.494,
          'grid_sampling':'nearest latitude/longitude grid cell, repeated grid cells retained as reef locations',
          'interpretation':'Annual-mean Eulerian signed meridional velocity; not residence time, particle pathways or settlement flux.'}
    (out/'environment_metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(meta));print(summary.to_string())

if __name__=='__main__':main()
