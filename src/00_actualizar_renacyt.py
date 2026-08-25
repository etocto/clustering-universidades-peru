
import sys,argparse,unicodedata,re
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd
from rapidfuzz import process,fuzz
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).parent))
from config import DATA_DIR,FEATURE_COLS
FECHA_CORTE=datetime(2026,4,16)
AÑOS_RECIENTE=2
UMBRAL_FUZZY=82
NIVEL_NUM={"I":7,"II":6,"III":5,"IV":4,"V":3,"VI":2,"VII":1,"Investigador Distinguido":8,"INVESTIGADOR DISTINGUIDO":8}
SIN_RENACYT={"Escuela Nacional Superior de Arte Dramático \"Guillermo Ugarte Chamorro\"","Escuela Superior de Arte Dramático \"Virgilio Rodríguez Nache\"","Escuela Nacional Superior de Folklore José María Arguedas","Escuela Superior de Formación Artística Sérvulo Gutiérrez Alarcón de Ica","Escuela Superior de Formación Artística Pública Mario Urteaga Alvarado de Cajamarca","Escuela Superior de Música Pública José María Valle Riestra de Piura","Escuela Nacional Superior de Ballet","Escuela Superior de Arte Pública Ignacio Merino de Piura","Instituto Superior de Música Público Daniel Alomía Robles de Huánuco","Instituto Superior de Música Público Leandro Alviña Miranda del Cusco","Conservatorio Regional de Música Luis Duncker Lavalle","Escuela Superior de Guerra Naval","Facultad de Teología Pontificia y Civil de Lima","Universidad Jaime Bausate y Meza","Universidad Privada de Pucallpa S.A.C."}
CUSTOM_SEARCH={"Universidad Nacional Tecnológica de Lima Sur":"TECNOLOGICA DE LIMA SUR","Asociación Civil Universidad de Ciencias y Humanidades":"CIENCIAS Y HUMANIDADES","Universidad de Ciencias y Artes de América Latina S.A.C.":"CIENCIAS Y ARTES DE AMERICA LATINA","Universidad para el Desarrollo Andino":"DESARROLLO ANDINO"}
def norm(name):
    if pd.isna(name): return ""
    n=str(name).upper().strip()
    n=unicodedata.normalize("NFKD",n).encode("ascii","ignore").decode("ascii")
    for s in ["S.A.C.","S.A. C.","S.A.","S.R.L.","ASOCIACION CIVIL"," O UTP S.A.C.","E.I.R.L.","SOCIEDAD ANONIMA CERRADA"," SAC"," SRL"]:
        n=n.replace(s," ")
    return re.sub(r"\s+"," ",n).strip()
def parse_fecha(s):
    if pd.isna(s): return None
    try: return datetime.strptime(str(s).strip(),"%d/%m/%Y")
    except: return None
def extraer_puntaje(cal_str):
    if pd.isna(cal_str): return None
    for entry in str(cal_str).split("|"):
        parts=entry.strip().split(",")
        if len(parts)>=4:
            try: return float(parts[3].strip())
            except: continue
    return None
def calcular_metricas(grupo,doc_total,col_nivel,col_gen,col_fing,col_fprod,col_ocde,col_cal):
    n=len(grupo)
    if n==0: return dict(n_renacyt=0,pct_renacyt_doc=0.0,puntaje_medio=0.0,nivel_medio=0.0,pct_fem_renacyt=0.0,pct_prod_rec=0.0,n_areas_ocde=0,antiguedad_med=0.0)
    puntajes=grupo[col_cal].apply(extraer_puntaje).dropna()
    puntaje_medio=round(puntajes.mean(),3) if len(puntajes)>0 else 0.0
    niveles=grupo[col_nivel].map(NIVEL_NUM).dropna()
    nivel_medio=round(niveles.mean(),3) if len(niveles)>0 else 0.0
    fem=(grupo[col_gen].str.upper()=="FEMENINO").sum()
    pct_fem=round(fem/n*100,3)
    fecha_lim=datetime(FECHA_CORTE.year-AÑOS_RECIENTE,FECHA_CORTE.month,FECHA_CORTE.day)
    fechas_prod=grupo[col_fprod].apply(parse_fecha).dropna()
    pct_prod=round((fechas_prod>=fecha_lim).sum()/n*100,3) if len(fechas_prod)>0 else 0.0
    areas=set()
    for val in grupo[col_ocde].dropna():
        for bloque in str(val).split("||"):
            top=bloque.split("|")[0].strip()
            if top and top.lower() not in ("n","nu","null",""): areas.add(top)
    fechas_ing=grupo[col_fing].apply(parse_fecha).dropna()
    antiguedad=round(float(np.mean([(FECHA_CORTE-f).days/365.25 for f in fechas_ing])),3) if len(fechas_ing)>0 else 0.0
    pct_doc=round(n/doc_total*100,3) if doc_total>0 else 0.0
    return dict(n_renacyt=n,pct_renacyt_doc=pct_doc,puntaje_medio=puntaje_medio,nivel_medio=nivel_medio,pct_fem_renacyt=pct_fem,pct_prod_rec=pct_prod,n_areas_ocde=len(areas),antiguedad_med=antiguedad)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reporte",default=str(DATA_DIR/"ReporteInvestigadores_1.xlsx"))
    args=parser.parse_args()
    rp=Path(args.reporte)
    if not rp.exists(): print(f"No encontrado: {rp}"); sys.exit(1)
    print(f"Cargando {rp.name} ...")
    df_raw=pd.read_excel(rp,header=1)
    col_cond=df_raw.columns[6]; col_inst=df_raw.columns[4]; col_nivel=df_raw.columns[8]
    col_gen=df_raw.columns[12]; col_fing=df_raw.columns[13]; col_fprod=df_raw.columns[14]
    col_ocde=df_raw.columns[10]; col_cal=df_raw.columns[15]
    df_act=df_raw[df_raw[col_cond]=="Activo"].copy()
    print(f"  Investigadores activos: {len(df_act):,}")
    df_act["_norm"]=df_act[col_inst].apply(norm)
    df_act["_upper"]=df_act[col_inst].str.upper().fillna("")
    mp=DATA_DIR/"matriz_maestra.csv"
    df_m=pd.read_csv(mp)
    print(f"  IES en proyecto: {len(df_m)}")
    norms_uniq=[n for n in df_act["_norm"].unique() if n]
    print("Mapeando instituciones...")
    mapeo={}
    for ies in df_m["universidad"].values:
        if ies in SIN_RENACYT: mapeo[ies]=("none",None); continue
        if ies in CUSTOM_SEARCH: mapeo[ies]=("contains",CUSTOM_SEARCH[ies]); continue
        ies_n=norm(ies)
        if ies_n in norms_uniq: mapeo[ies]=("norm",ies_n); continue
        res=process.extractOne(ies_n,norms_uniq,scorer=fuzz.token_sort_ratio)
        if res and res[1]>=UMBRAL_FUZZY: mapeo[ies]=("norm",res[0])
        else: mapeo[ies]=("none",None); print(f"  Sin match: {ies}")
    ok=sum(1 for v in mapeo.values() if v[0]!="none")
    print(f"  Matches validos: {ok}/99")
    print("\nCalculando metricas RENACYT por IES...")
    resultados=[]
    for _,row in df_m.iterrows():
        ies=row["universidad"]; doc_t=int(row.get("doc_total",1))
        modo,valor=mapeo.get(ies,("none",None))
        if modo=="norm": grupo=df_act[df_act["_norm"]==valor]
        elif modo=="contains": grupo=df_act[df_act["_upper"].str.contains(valor,na=False)]
        else: grupo=pd.DataFrame()
        resultados.append(calcular_metricas(grupo,doc_t,col_nivel,col_gen,col_fing,col_fprod,col_ocde,col_cal))
    df_res=pd.DataFrame(resultados,index=df_m.index)
    print(f"\n{chr(39)}Universidad{chr(39):<50} {chr(39)}Antes{chr(39):>7} {chr(39)}Despues{chr(39):>9}")
    print("-"*70)
    for i,row in df_m.iterrows():
        ant=int(row.get("n_renacyt",0)); desp=int(df_res.loc[i,"n_renacyt"])
        modo,_=mapeo.get(row["universidad"],("none",None))
        flag="OK  " if desp>=ant else "BAJA"
        print(f"  [{flag}] {row['universidad'][:46]:<48} {ant:>7} {desp:>9}")
    for var in df_res.columns:
        if var in df_m.columns: df_m[var]=df_res[var].values
    df_m.to_csv(mp,index=False)
    print(f"\nGuardado: {mp.name}")
    print("Regenerando matriz_escalada.csv ...")
    ep=DATA_DIR/"matriz_escalada.csv"
    X=df_m[FEATURE_COLS].values.astype(float)
    df_e=pd.DataFrame(StandardScaler().fit_transform(X),columns=FEATURE_COLS)
    df_e.insert(0,"universidad",df_m["universidad"].values)
    df_e.to_csv(ep,index=False)
    print(f"Guardado: {ep.name}")
    print("\nEjecuta ahora el pipeline completo:")
    for s in ["01_pca","02_k_optimo","03_clustering","04_perfilado","05_longitudinal","06_regional","07_disciplinar","regenerar_figuras_siglas"]:
        print(f"  python src/{s}.py")
if __name__=="__main__": main()

