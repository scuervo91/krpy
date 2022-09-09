import pandas as pd
import numpy as np
from pydantic import BaseModel, Field, validator, validate_arguments
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from typing import Optional, Union, List, Dict
import matplotlib.pyplot as plt

def kr_curve(sn:np.ndarray, n:float, krend:float) -> np.ndarray:
    """kr_curve [Estimate Relative permeability curve from normalized saturation, exponent and end-points]
    Parameters
    ----------
    sn : np.ndarray
        [Saturation Array]
    n : float
        [Exponent]
    krend : float
        [End-point]
    Returns
    -------
    np.ndarray
        [Relative permeability curve]
    """
    return krend * np.power(sn,n)

def sw_normalize(sw:np.ndarray, swir:float, sor:float, sgc:float=0) -> np.ndarray:
    """sw_normalize [Convert array of water saturation to normalized water saturation]
    Parameters
    ----------
    sw : np.ndarray
        [Water saturation array]
    swir : float
        [Irreducible water saturation]
    sor : float
        [Residual Oil Saturation]
    Returns
    -------
    np.ndarray
        [Normalized water Saturation]
    """
    swn = (sw - swir) / (1 - swir - sor - sgc)
    return swn

def sg_normalize(sg:np.ndarray, swir:float, sor:float, sgc:float=0) -> np.ndarray:
    """sg_normalize [Convert array of water saturation to normalized water saturation]
    Parameters
    ----------
    sw : np.ndarray
        [Water saturation array]
    swir : float
        [Irreducible water saturation]
    sor : float
        [Residual Oil Saturation]
    Returns
    -------
    np.ndarray
        [Normalized water Saturation]
    """
    sgn = (sg - sgc) / (1 - swir - sor - sgc)
    return sgn

def sw_denormalize(swn:np.ndarray, swir:float, sor:float,sgc:float=0) -> np.ndarray:
    """sw_normalize [Convert array of normalized water saturation to water saturation]
    Parameters
    ----------
    sw : np.ndarray
        [normalized Water saturation array]
    swir : float
        [Irreducible water saturation]
    sor : float
        [Residual Oil Saturation]
    Returns
    -------
    np.ndarray
        [water Saturation]
    """
    sw = swn * (1 - swir - sor - sgc) + swir 
    return sw

def sg_denormalize(sgn:np.ndarray, swir:float, sor:float,sgc:float=0) -> np.ndarray:
    """sg_normalize [Convert array of normalized water saturation to water saturation]
    Parameters
    ----------
    sw : np.ndarray
        [normalized Water saturation array]
    swir : float
        [Irreducible water saturation]
    sor : float
        [Residual Oil Saturation]
    Returns
    -------
    np.ndarray
        [water Saturation]
    """
    sg = sgn * (1 - swir - sor - sgc) + sgc
    return sg

class Kr(BaseModel):
    saturation: Optional[Union[List[float], np.ndarray]] = Field(None)
    fields: Optional[Dict[str,Union[List[float], np.ndarray]]] = Field(None)
    swir: Optional[float] = Field(None, description='Irreducible water saturation')
    sor: Optional[float] = Field(None, description='Residual oil saturation')
    sgc: Optional[float] = Field(None, description='Residual gas saturation')

    @validator('saturation')
    def check_array_saturation_order(cls, v):
        v = np.atleast_1d(v)
        diff = np.diff(np.array(v))
        if not any([np.all(diff>0),np.all(diff<0)]):
            raise ValueError('Saturation must be ordered')
        return v

    class Config:
        extra = 'forbid'
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {np.ndarray: lambda x: x.tolist()}

    def df(self):
        d = self.dict()
        _df = pd.DataFrame(d['fields'], index=d['saturation'])
        _df.index.name = 'saturation'
        
        return _df
    
    def interpolate(self, value, cols=None):
        p = np.atleast_1d(value)
        
        int_dict={}
        
        int_cols = list(self.fields.keys()) if cols is None else cols
        
        for i in int_cols:
            int_dict[i] = interp1d(self.saturation,self.fields[i],bounds_error=False,fill_value='extrapolate')(p)

        int_df = pd.DataFrame(int_dict, index=p)
        int_df.index.name = 'saturation'
        return int_df 
    
########### KROW #############

class Corey(BaseModel):
    nw: float = Field(2., description='Exponent for water saturation')
    no: float = Field(2., description='Exponent for oil saturation')
    ng: float = Field(2., description='Exponent for gas saturation')
    npcwo: float = Field(2., description='Exponent for capillary pressure Oil water')
    npcog: float = Field(2., description='Exponent for capillary pressure Oil Gas')
    krw_end: float = Field(1., gt=0, le=1, description='End-point for water relative permeability')
    kro_end: float = Field(1., gt=0, le=1, description='End-point for oil relative permeability')
    krg_end: float = Field(1., gt=0, le=1, description='End-point for gas relative permeability')
    pco_end: float = Field(0., description='End-point for capillary pressure')
    pcg_end: float = Field(0., description='End-point for capillary pressure')
    pdwo: float = Field(0., description='Drainage pressure Oil Water')
    pdog: float = Field(0., description='Drainage pressure Oil Gas')
    
    @classmethod
    def fit(cls, df, sw:str='sw', krw:str='krw', kro:str='kro', swir:float=None, sor:float=None, guess_krw = None, guess_kro=None):
        
        if sw is None:
            sw_array = df.index.values 
        else:
            sw_array = df[sw].values
            
        if swir is None:
            swir = df.loc[df[krw]==0,sw].max()
        if sor is None:
            sor = df.loc[df[kro]==0,sw].min()
            
        swn = sw_normalize(sw_array, swir, sor)
        d = {}
        
        if krw is not None:
            krw_array = df[krw].values
        
            popt, pcov = curve_fit(kr_curve, swn, krw_array, bounds=([0.0,0], [np.inf, 1]), p0=guess_krw)
            
            d['nw'] = popt[0]
            d['krw_end'] = popt[1]
            
        if kro is not None:
            kro_array = df[kro].values
        
            popt, pcov = curve_fit(kr_curve, 1-swn, kro_array, bounds=([0.0,0], [np.inf, 1]),p0=guess_kro)

            d['no'] = popt[0]
            d['kro_end'] = popt[1]
            
        return cls(**d)

class Krog(Kr):
    
    @classmethod
    def from_corey(
        cls,
        corey:Corey,
        swir=None,
        sor=None,
        sgc=0,
        n:int=10
    ):

        sgn = np.linspace(0,1,n)
        son = 1 - sgn
        
        kro = kr_curve(son, corey.no, corey.kro_end)
        krg = kr_curve(sgn, corey.ng, corey.krg_end)
        pcog = (corey.pcg_end * np.power(son,corey.npcog)) + corey.pdog
        
        kr_ratio = krg / kro
        
        sg = sg_denormalize(sgn, swir, sor,sgc=sgc)

        dict_krs = {
                'krg': krg,
                'kro': kro,
                'pcog': pcog,
                'sgn': sgn,
                'kr_ratio': kr_ratio
            }    
        return cls(
            saturation = sg,
            swir = swir,
            sor = sor,
            sgc = sgc,
            fields = dict_krs
        )

    def to_ecl(self,krg = 'krg',kro = 'kro',pcog = 'pcog'):
        
        df = self.df()[[krg,kro,pcog]]
        string = "SGOF\n"
        
        string += df.reset_index().to_string(header=False, index=False) +'/\n'
        
        return string
    
    def plot(
        self,
        krg = 'krg',
        kro = 'kro',
        pcog = 'pcog',        
        ax=None,
        norm=False, 
        ann=False, 
        pc = False,
        kr = True,
        krg_kw={},
        kro_kw={}, 
        ann_kw = {},
        pcog_kw = {}
    ):

        df = self.df()
        
        ax_list = []
        #Set default plot properties krg
        def_krg_kw = {
            'color': 'red',
            'linestyle':'--',
            'linewidth': 2
            }    

        for (k,v) in def_krg_kw.items():
            if k not in krg_kw:
                krg_kw[k]=v

        #Set default plot properties kro
        def_kro_kw = {
            'color': 'green',
            'linestyle':'--',
            'linewidth': 2
            }    

        for (k,v) in def_kro_kw.items():
            if k not in kro_kw:
                kro_kw[k]=v

        def_pc_kw = {
            'color': 'black',
            'linestyle':'--',
            'linewidth': 1
            }    

        for (k,v) in def_pc_kw.items():
            if k not in pcog_kw:
                pcog_kw[k]=v

        #Set default plot properties kro
        def_ann_kw = {
            'xytext': (0,15),
            'textcoords':'offset points',
            'arrowprops': {'arrowstyle':"->"},
            'bbox':{'boxstyle':'round', 'fc':'0.8'},
            'fontsize':11
            }    

        for (k,v) in def_ann_kw.items():
            if k not in ann_kw:
                ann_kw[k]=v


        if kr:
            if norm:
                sg_x = sg_normalize(df.index, self.swir, self.sor, sgc=self.sgc)
                krg = df[krg].values
                kro = df[kro].values
            else:
                sg_x = df.index 
                krg = df[krg].values
                kro = df[kro].values

            #Set the axes      
            krax= ax or plt.gca()
            krax.plot(sg_x, krg, **krg_kw)
            krax.plot(sg_x, kro, **kro_kw)

            #set labels
            krax.set_xlabel('Gas Saturation []')
            krax.set_ylabel('Kr []')
            krax.set_xlim([0,1])
            krax.set_ylim([0,1])
            ax_list.append(krax)

          
        #Annotate
        if ann and kr and not norm:
            krax.annotate(
                'sgir',
                xy = (df.index[0],df[krg].iloc[0]),
                xycoords='data',
                **ann_kw
            ) 
            krax.annotate(
                'sgor',
                xy = (df.index[-1],df[kro].iloc[-1]),
                xycoords='data',
                **ann_kw
            ) 
            krax.annotate(
                'kroend',
                xy = (df.index[0],df[kro].iloc[0]),
                xycoords='data',
                **ann_kw
            ) 
            krax.annotate(
                'krgend',
                xy = (df.index[-1],df[krg].iloc[-1]),
                xycoords='data',
                **ann_kw
            ) 

        
        
        if pc and not norm:
            if kr:
                pcax=krax.twinx()
                pcax.yaxis.set_label_position("right")
                pcax.yaxis.tick_right()
            else:
                pcax= ax or plt.gca()

            pcax.plot(df.index, df[pcog], **pcog_kw)
            pcax.set_ylabel('Capillary Pressure [psi]')
            
            ax_list.append(pcax)
        
        return ax_list
        
        
class Krow(Kr):

    @classmethod
    def from_corey(cls, corey:Corey, swir = None, sor = None, sgc = 0, n:int=10):
               
        swn = np.linspace(0,1,n)
        son = 1 - swn
        
        kro = kr_curve(son, corey.no, corey.kro_end)
        krw = kr_curve(swn, corey.nw, corey.krw_end)
        pcwo = (corey.pco_end * np.power(son,corey.npcwo)) + corey.pdwo
        
        kr_ratio = kro / krw
        
        sw = sw_denormalize(swn, swir, sor,sgc=sgc)

        dict_krs = {
                'krw': krw,
                'kro': kro,
                'pcwo': pcwo,
                'swn': swn,
                'kr_ratio': kr_ratio
            }    
        return cls(
            saturation = sw,
            swir = swir,
            sor = sor,
            sgc = sgc,
            fields = dict_krs
        )

    
    def plot(
        self,
        krw = 'krw',
        kro = 'kro',
        pcwo = 'pcwo',        
        ax=None,
        norm=False, 
        ann=False, 
        pc = False,
        kr = True,
        krw_kw={},
        kro_kw={}, 
        ann_kw = {},
        pcwo_kw = {}
    ):

        df = self.df()
        
        ax_list = []
        #Set default plot properties krw
        def_krw_kw = {
            'color': 'blue',
            'linestyle':'--',
            'linewidth': 2
            }    

        for (k,v) in def_krw_kw.items():
            if k not in krw_kw:
                krw_kw[k]=v

        #Set default plot properties kro
        def_kro_kw = {
            'color': 'green',
            'linestyle':'--',
            'linewidth': 2
            }    

        for (k,v) in def_kro_kw.items():
            if k not in kro_kw:
                kro_kw[k]=v

        def_pc_kw = {
            'color': 'black',
            'linestyle':'--',
            'linewidth': 1
            }    

        for (k,v) in def_pc_kw.items():
            if k not in pcwo_kw:
                pcwo_kw[k]=v

        #Set default plot properties kro
        def_ann_kw = {
            'xytext': (0,15),
            'textcoords':'offset points',
            'arrowprops': {'arrowstyle':"->"},
            'bbox':{'boxstyle':'round', 'fc':'0.8'},
            'fontsize':11
            }    

        for (k,v) in def_ann_kw.items():
            if k not in ann_kw:
                ann_kw[k]=v


        if kr:
            if norm:
                sw_x = sw_normalize(df.index, self.swir, self.sor)
                krw = df[krw].values
                kro = df[kro].values
            else:
                sw_x = df.index 
                krw = df[krw].values
                kro = df[kro].values

            #Set the axes      
            krax= ax or plt.gca()
            krax.plot(sw_x, krw, **krw_kw)
            krax.plot(sw_x, kro, **kro_kw)

            #set labels
            krax.set_xlabel('Water Saturation []')
            krax.set_ylabel('Kr []')
            krax.set_xlim([0,1])
            krax.set_ylim([0,1])
            ax_list.append(krax)

          
        #Annotate
        if ann and kr and not norm:
            krax.annotate(
                'swir',
                xy = (df.index[0],df[krw].iloc[0]),
                xycoords='data',
                **ann_kw
            ) 
            krax.annotate(
                'swor',
                xy = (df.index[-1],df[kro].iloc[-1]),
                xycoords='data',
                **ann_kw
            ) 
            krax.annotate(
                'kroend',
                xy = (df.index[0],df[kro].iloc[0]),
                xycoords='data',
                **ann_kw
            ) 
            krax.annotate(
                'krwend',
                xy = (df.index[-1],df[krw].iloc[-1]),
                xycoords='data',
                **ann_kw
            ) 

        
        
        if pc and not norm:
            if kr:
                pcax=krax.twinx()
                pcax.yaxis.set_label_position("right")
                pcax.yaxis.tick_right()
            else:
                pcax= ax or plt.gca()

            pcax.plot(df.index, df[pcwo], **pcwo_kw)
            pcax.set_ylabel('Capillary Pressure [psi]')
            
            ax_list.append(pcax)
        
        return ax_list
        
    
    def to_ecl(self,krw = 'krw',kro = 'kro',pcwo = 'pcwo'):
        
        df = self.df()[[krw,kro,pcwo]]
        string = "SWOF\n"
        
        string += df.reset_index().to_string(header=False, index=False) +'/\n'
        
        return string
    

    
    def get_height(self, rhoo:float, rhow:float = 62.28, pcwo:str='pcwo'):
        df = self.df()
        
        if pcwo not in df.columns:
            raise ValueError('pcwo not in df')
        
        delta_rho = rhow - rhoo
        h = (144*df[pcwo])/delta_rho
        
        self.fields['height'] = h
        
        return self