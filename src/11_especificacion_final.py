import sys, numpy as np, pandas as pd
sys.path.insert(0,'src')
from config import DATA_DIR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.stats import spearmanr

EMP=["pct_doctorado","pct_maestria","pct_exclusiva","pct_tc","pct_contratado","pct_ordinario","edad_media_doc","pct_fem_doc"]
ENR=["pct_fem_mat","pct_discap","pct_posgrado","edad_media_mat","n_departamentos","nota_prom_egr","creditos_prom_egr","pct_posgrado_egr","ratio_mat_doc","ratio_egr_mat"]
S2 = EMP+ENR+["has_renacyt","pct_renacyt_doc"]

df=pd.read_csv(DATA_DIR/'matriz_maestra.csv')
df['has_renacyt']=(df.n_renacyt>0).astype(int)
base=np.load(DATA_DIR/'labels_final.npy')+1
df['cl_pub']=base
c3=[c for c in np.unique(base) if (base==c).sum()==3][0]
pg=df.cl_pub==c3

# --- WINSORIZACION ---
aff=df[df.pct_renacyt_doc>100]
print("Winsorizadas al 100% (mas investigadores RENACYT que docentes reportados):")
print(aff[['universidad','doc_total','n_renacyt','pct_renacyt_doc']].to_string(index=False))
df['pct_renacyt_doc_w']=df.pct_renacyt_doc.clip(upper=100)
S2w=[c if c!='pct_renacyt_doc' else 'pct_renacyt_doc_w' for c in S2]

def fit(cols,data,k):
    X=StandardScaler().fit_transform(data[cols].fillna(0).values)
    p=PCA(random_state=42).fit(X); cum=np.cumsum(p.explained_variance_ratio_)
    nc=int(np.argmax(cum>=0.90)+1)
    Xp=PCA(n_components=nc,random_state=42).fit_transform(X)
    km=KMeans(n_clusters=k,init='k-means++',n_init=50,max_iter=500,random_state=42).fit(Xp)
    return km.labels_+1,Xp,nc,cum[nc-1],p.explained_variance_ratio_[:2]

sub=df[~pg].reset_index(drop=True)
print("\nSeleccion de k (S2 winsorizada, n=96):")
for k in range(2,8):
    lab,Xp,nc,cv,pcs=fit(S2w,sub,k)
    print(f"  k={k}: ASW={silhouette_score(Xp,lab):.3f}  tamanos={sorted(np.bincount(lab-1))[::-1]}")

lab,Xp,nc,cv,pcs=fit(S2w,sub,3)
sub['P']=lab
print(f"\n=== ESPECIFICACION FINAL: S2 winsorizada, estrato a priori, k=3 ===")
print(f"n={len(sub)}  vars={len(S2w)}  componentes={nc} ({cv*100:.1f}%)  PC1={pcs[0]*100:.1f}%  PC2={pcs[1]*100:.1f}%")
print(f"ASW={silhouette_score(Xp,lab):.3f}  ARI vs tipologia publicada={adjusted_rand_score(sub.cl_pub,lab):.3f}")
lab_nw,Xnw,_,_,_=fit(S2,sub,3)
print(f"ARI winsorizada vs sin winsorizar={adjusted_rand_score(lab_nw,lab):.3f}")
print()
print(pd.crosstab(sub.cl_pub,sub.P,rownames=['publicado'],colnames=['perfil']).to_string())
prof=sub.groupby('P').agg(n=('universidad','size'),
    contratado=('pct_contratado','mean'), ordinario=('pct_ordinario','mean'),
    renacyt=('pct_renacyt_doc_w','mean'), sin_renacyt=('has_renacyt',lambda s:100*(1-s.mean())),
    publico=('es_publico','mean'), universidad=('es_universidad','mean')).round(1)
prof['publico']*=100; prof['universidad']*=100
print(); print(prof.to_string())

con=df.has_renacyt==1
print("\nAsociacion empleo x acoplamiento (Spearman, n=84 con RENACYT, winsorizada):")
for ev in ["pct_ordinario","pct_contratado","pct_tc"]:
    r,p=spearmanr(df.loc[con,ev],df.loc[con,'pct_renacyt_doc_w'])
    print(f"  {ev:16s} rho={r:+.2f}  p={p:.4f}")
