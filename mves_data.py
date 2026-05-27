"""
mves_data.py — Módulo de datos MVES-CO (versión final)
Sistema de Alerta Temprana: Contagio Financiero Intersectorial Colombia
Maestría en Analítica de Datos · Universidad Central · 2026
Autores: Katy Pacheco Manchego · Julian Andres Firacative Varon
"""
import pandas as pd
import numpy as np
import json
import streamlit as st
from pathlib import Path

MACROS = {
    "M01":"Agropecuario","M02":"Minería y energía","M03":"Manufactura",
    "M04":"Electricidad y agua","M05":"Construcción",
    "M06":"Comercio, transporte y turismo","M07":"TIC y economía digital",
    "M08":"Financiero y seguros","M09":"Inmobiliario",
    "M10":"Servicios profesionales","M11":"Gobierno, educación y salud",
    "M12":"Arte y recreación",
}
MACROS_LIST = list(MACROS.keys())
COLORES_ESTADO = {"S1":"#639922","S2":"#1D9E75","S3":"#378ADD","S4":"#BA7517","S5":"#E24B4A"}
NOMBRES_ESTADO = {"S1":"Aceleración","S2":"Crecimiento","S3":"Estabilidad",
                  "S4":"Decrecimiento","S5":"Contracción"}
COLORES_REGIMEN = {0:"#639922",1:"#378ADD",2:"#E24B4A"}
NOMBRES_REGIMEN = {0:"R1 Expansión",1:"R2 Estabilidad",2:"R3 Contracción"}
COLORES_ALERTA  = {"VERDE":"#639922","AMARILLA":"#BA7517","ROJA":"#E24B4A"}
PESOS_PIB = {"M01":0.068,"M02":0.072,"M03":0.121,"M04":0.038,"M05":0.062,
             "M06":0.187,"M07":0.035,"M08":0.052,"M09":0.091,"M10":0.081,"M11":0.158,"M12":0.035}
DATOS_DIR = Path(__file__).parent / "datos"

@st.cache_data
def cargar_dashboard_data():
    with open(DATOS_DIR / "dashboard_data.json") as f:
        return json.load(f)

@st.cache_data
def cargar_panel():
    return pd.read_csv(DATOS_DIR / "icds_mensual_definitivo.csv").sort_values(["macrosector_id","fecha"]).reset_index(drop=True)

@st.cache_data
def cargar_icds_star():
    return pd.read_csv(DATOS_DIR / "icds_star_final.csv").sort_values(["macrosector_id","fecha"]).reset_index(drop=True)

@st.cache_data
def cargar_icds_agg():
    return pd.read_csv(DATOS_DIR / "icds_agg_final.csv").sort_values("fecha").reset_index(drop=True)

@st.cache_data
def cargar_markov():
    df = pd.read_csv(DATOS_DIR / "markov_resultados.csv")
    df["Periodo"] = pd.to_datetime(df["Periodo"], errors="coerce")
    return df.sort_values("Periodo").reset_index(drop=True)

@st.cache_data
def cargar_markov_params():
    with open(DATOS_DIR / "markov_params.json") as f:
        return json.load(f)

@st.cache_data
def cargar_leontief():
    with open(DATOS_DIR / "leontief_macro.json") as f:
        d = json.load(f)
    return pd.DataFrame(d["L"]), pd.DataFrame(d["A"])

def simular_choque(df_star, A, sector_crisis, icds_crisis=0.10):
    ult = df_star.dropna(subset=["icds_star"])["fecha"].max()
    base = df_star[df_star["fecha"]==ult].set_index("macrosector_id")["icds"].to_dict()
    shock = {**base, sector_crisis: icds_crisis}
    rows = []
    for i in MACROS_LIST:
        b = base.get(i, 0.5)
        pen = sum(float(A.loc[i,j])*max(0,0.5-shock.get(j,0.5)) for j in MACROS_LIST)
        rows.append({"macrosector_id":i,"macrosector":MACROS[i],
                     "icds_base":round(float(b),4),
                     "icds_shock":round(max(0.0,float(b)-0.5*pen),4),
                     "caida":round(float(b)-max(0.0,float(b)-0.5*pen),4)})
    return pd.DataFrame(rows).sort_values("caida",ascending=False)
