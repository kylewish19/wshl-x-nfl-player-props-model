"""Shadow-only empirical workload-conditioned yardage probabilities."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .models import TrainedPropModel

def workload_keys(X, volume, bands, stability):
    avg=pd.to_numeric(X[volume+"_r3"],errors="coerce").to_numpy(dtype=float)
    lag=pd.to_numeric(X[volume+"_lag1"],errors="coerce").to_numpy(dtype=float)
    out=[]
    for a,l in zip(avg,lag):
        if not np.isfinite(a) or (stability and not np.isfinite(l)):
            out.append(None); continue
        band=int(np.digitize(a,bands))
        unstable=int(abs(l-a)/max(a,1)>=.5)
        out.append((band,unstable) if stability else (band,))
    return out

def tail(residuals, threshold):
    r=np.sort(np.asarray(residuals,dtype=float))
    if len(r)<20 or not np.isfinite(r).all():
        raise ValueError("Insufficient finite calibration residuals")
    return 1-np.searchsorted(r,threshold,side="right")/len(r)

@dataclass
class WorkloadShadow(TrainedPropModel):
    volume: str
    bands: list[int]
    stability: bool
    groups: dict
    min_group: int=20
    shrink: int=100

    def pooled_probability_over(self,X,line):
        return tail(self.residuals,line-self.predict_point(X))

    def probability_over(self,X,line):
        threshold=line-self.predict_point(X)
        pooled=tail(self.residuals,threshold)
        result=pooled.copy()
        for i,key in enumerate(workload_keys(X,self.volume,self.bands,self.stability)):
            r=self.groups.get(key,[])
            n=len(r)
            if n>=self.min_group:
                w=n/(n+self.shrink)
                result[i]=w*tail(r,threshold[i])+(1-w)*pooled[i]
        return result
