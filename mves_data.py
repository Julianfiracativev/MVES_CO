"""
mves_data.py — Carga y procesamiento de datos compartidos para el dashboard MVES-CO
Maestría en Analítica de Datos · Universidad Central · 2026
"""
import pandas as pd
import numpy as np
import json
import streamlit as st
from pathlib import Path
import networkx as nx

# ── Constantes ─────────────────────────────────────────────────────────────────
MACROS = {
    "M01": "Agropecuario",
    "M02": "Minería y energía",
    "M03": "Manufactura",
    "M04": "Electricidad y agua",
    "M05": "Construcción",
    "M06": "Comercio, transporte y turismo",
    "M07": "TIC y economía digital",
    "M08": "Financiero y seguros",
    "M09": "Inmobiliario",
    "M10": "Servicios profesionales",
    "M11": "Gobierno, educación y salud",
    "M12": "Arte y recreación",
}
MACROS_LIST = list(MACROS.keys())

COLORES_ESTADO = {
    "S1": "#639922", "S2": "#1D9E75", "S3": "#378ADD",
    "S4": "#BA7517", "S5": "#E24B4A",
}
NOMBRES_ESTADO = {
    "S1": "Aceleración", "S2": "Crecimiento", "S3": "Estabilidad",
    "S4": "Decrecimiento", "S5": "Contracción",
}
COLORES_REGIMEN = {0: "#639922", 1: "#378ADD", 2: "#E24B4A"}
NOMBRES_REGIMEN = {0: "R1 Expansión", 1: "R2 Estabilidad", 2: "R3 Contracción"}

PESOS_PIB = {
    "M01": 0.068, "M02": 0.072, "M03": 0.121, "M04": 0.038, "M05": 0.062,
    "M06": 0.187, "M07": 0.035, "M08": 0.052, "M09": 0.091,
    "M10": 0.081, "M11": 0.158, "M12": 0.035,
}

EXCLUIR_COVID = (
    [f"2020-{m:02d}" for m in range(1, 13)] +
    [f"2021-{m:02d}" for m in range(1, 13)]
)

DATOS_DIR = Path(__file__).parent / "datos"


# ── Carga con caché ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_panel() -> pd.DataFrame:
    df = pd.read_csv(DATOS_DIR / "icds_mensual_definitivo.csv")
    return df.sort_values(["macrosector_id", "fecha"]).reset_index(drop=True)


@st.cache_data
def cargar_leontief():
    with open(DATOS_DIR / "leontief_macro.json") as f:
        d = json.load(f)
    L = pd.DataFrame(d["L"])
    A = pd.DataFrame(d["A"])
    return L, A


@st.cache_data
def calcular_icds_star(panel: pd.DataFrame, A: pd.DataFrame) -> pd.DataFrame:
    """ICDS* ajustado por contagio MIP."""
    UMBRALES_CLS = [(0.80, "S1"), (0.60, "S2"), (0.40, "S3"), (0.20, "S4"), (0.00, "S5")]
    def cls(v):
        if pd.isna(v): return None
        for u, c in UMBRALES_CLS:
            if v >= u: return c
        return "S5"

    rows = []
    for fecha in sorted(panel["fecha"].unique()):
        sub = panel[panel["fecha"] == fecha].set_index("macrosector_id")
        icds_t = {m: float(sub.loc[m, "icds"]) if m in sub.index and not pd.isna(sub.loc[m, "icds"]) else 0.5
                  for m in MACROS_LIST}
        for i in MACROS_LIST:
            base = icds_t[i]
            pen = sum(float(A.loc[i, j]) * max(0, 0.5 - icds_t.get(j, 0.5)) for j in MACROS_LIST)
            star = round(max(0.0, base - 0.5 * pen), 4)
            if i in sub.index:
                r = sub.loc[i].to_dict()
                r.update({"fecha": fecha, "macrosector_id": i,
                          "macrosector": MACROS[i],
                          "icds_star": star,
                          "ajuste_mip": round(base - star, 4),
                          "estado_star": cls(star)})
                rows.append(r)
    return pd.DataFrame(rows)


@st.cache_data
def calcular_serie_agregada(df_star: pd.DataFrame):
    """ICDS* agregado ponderado por PIB."""
    agg = (df_star.dropna(subset=["icds_star"])
           .assign(peso=lambda d: d["macrosector_id"].map(PESOS_PIB).fillna(1 / 12))
           .groupby("fecha")
           .apply(lambda g: np.average(g["icds_star"], weights=g["peso"]))
           .reset_index(name="icds_agg"))
    agg["icds_agg"] = agg["icds_agg"].round(4)
    agg = agg.sort_values("fecha").reset_index(drop=True)

    def clf_r(v):
        if v >= 0.60: return 0
        if v >= 0.40: return 1
        return 2

    agg["regimen"] = agg["icds_agg"].apply(clf_r)
    return agg


@st.cache_data
def calcular_transicion(serie_reg: np.ndarray, k: int = 3):
    P = np.zeros((k, k))
    for t in range(1, len(serie_reg)):
        P[serie_reg[t - 1], serie_reg[t]] += 1
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    return P / rs


def simular_choque(df_star: pd.DataFrame, A: pd.DataFrame,
                   sector_crisis: str, icds_crisis: float = 0.10) -> pd.DataFrame:
    ult = df_star.dropna(subset=["icds_star"])["fecha"].max()
    base = df_star[df_star["fecha"] == ult].set_index("macrosector_id")["icds"].to_dict()
    shock = {**base, sector_crisis: icds_crisis}

    rows = []
    for i in MACROS_LIST:
        b = base.get(i, 0.5)
        pen = sum(float(A.loc[i, j]) * max(0, 0.5 - shock.get(j, 0.5)) for j in MACROS_LIST)
        post = round(max(0.0, b - 0.5 * pen), 4)
        rows.append({
            "macrosector_id": i, "macrosector": MACROS[i],
            "icds_base": round(float(b), 4),
            "icds_shock": post,
            "caida": round(float(b) - post, 4),
        })
    return pd.DataFrame(rows).sort_values("caida", ascending=False)

# ── Modelo nuevo 1: Leontief + Red con base DANE limpia ─────────────────────
@st.cache_data
def cargar_base_dane_limpia(nombre_archivo: str = "base_dane_limpia.xlsx") -> pd.DataFrame:
    """Carga la matriz DANE limpia desde datos/. Soporta primera columna como índice."""
    path = DATOS_DIR / nombre_archivo
    Z = pd.read_excel(path, index_col=0)
    Z = Z.apply(pd.to_numeric, errors="coerce").fillna(0)
    Z.index = Z.index.astype(str).str.strip()
    Z.columns = Z.columns.astype(str).str.strip()
    comunes = [s for s in Z.index if s in list(Z.columns)]
    if not comunes:
        raise ValueError("No hay sectores comunes entre filas y columnas. Revisa la matriz base_dane_limpia.xlsx")
    Z = Z.loc[comunes, comunes]
    Z = Z.loc[Z.sum(axis=1) > 0, :]
    Z = Z.loc[:, Z.sum(axis=0) > 0]
    comunes_final = [s for s in Z.index if s in list(Z.columns)]
    Z = Z.loc[comunes_final, comunes_final]
    return Z

@st.cache_data
def ejecutar_modelo_leontief_nuevo(percentil_umbral: int = 85):
    """Calcula A, L, multiplicadores, red y ranking de criticidad para la base DANE nueva."""
    Z = cargar_base_dane_limpia()
    produccion_total = Z.sum(axis=0).replace(0, np.nan)
    A = Z.div(produccion_total, axis=1).replace([np.inf, -np.inf], 0).fillna(0)
    I = np.eye(A.shape[0])
    try:
        L_array = np.linalg.inv(I - A.values)
        metodo_inversa = "Inversa clásica"
    except np.linalg.LinAlgError:
        L_array = np.linalg.pinv(I - A.values)
        metodo_inversa = "Pseudo-inversa"
    L = pd.DataFrame(L_array, index=A.index, columns=A.columns)
    multiplicador_total = L.sum(axis=0)
    multiplicadores = pd.DataFrame({
        "Sector": L.columns,
        "Multiplicador_Total": multiplicador_total.values,
        "Multiplicador_Impacto": (multiplicador_total - 1).values,
    }).sort_values("Multiplicador_Impacto", ascending=False)

    valores = [float(L.loc[o, d]) for o in L.index for d in L.columns if o != d]
    umbral = float(np.percentile(valores, percentil_umbral)) if valores else 0.0
    G = nx.DiGraph()
    for _, row in multiplicadores.iterrows():
        G.add_node(row["Sector"], multiplicador=float(row["Multiplicador_Impacto"]))
    edges = []
    for origen in L.index:
        for destino in L.columns:
            if origen != destino:
                peso = float(L.loc[origen, destino])
                if peso > umbral:
                    G.add_edge(origen, destino, weight=peso)
                    edges.append({"Sector_Origen": origen, "Sector_Destino": destino, "Peso_Leontief": peso})

    if G.number_of_edges() > 0:
        pagerank = nx.pagerank(G, weight="weight")
        betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
        closeness = nx.closeness_centrality(G)
    else:
        pagerank = {n: 0 for n in G.nodes()}
        betweenness = {n: 0 for n in G.nodes()}
        closeness = {n: 0 for n in G.nodes()}
    degree = dict(G.degree())
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    metricas = pd.DataFrame({
        "Sector": list(G.nodes()),
        "Degree": [degree[n] for n in G.nodes()],
        "In_Degree": [in_degree[n] for n in G.nodes()],
        "Out_Degree": [out_degree[n] for n in G.nodes()],
        "PageRank": [pagerank[n] for n in G.nodes()],
        "Betweenness": [betweenness[n] for n in G.nodes()],
        "Closeness": [closeness[n] for n in G.nodes()],
    }).merge(multiplicadores, on="Sector", how="left")
    metricas["Criticidad"] = metricas["PageRank"] * metricas["Multiplicador_Impacto"]
    q80 = metricas["Criticidad"].quantile(0.80)
    q50 = metricas["Criticidad"].quantile(0.50)
    metricas["Alerta"] = np.where(metricas["Criticidad"] >= q80, "ALERTA ROJA",
                           np.where(metricas["Criticidad"] >= q50, "ALERTA AMARILLA", "ALERTA VERDE"))
    metricas = metricas.sort_values("Criticidad", ascending=False).reset_index(drop=True)
    aristas = pd.DataFrame(edges).sort_values("Peso_Leontief", ascending=False) if edges else pd.DataFrame(columns=["Sector_Origen","Sector_Destino","Peso_Leontief"])
    resumen = {
        "n_sectores": int(Z.shape[0]),
        "n_aristas": int(G.number_of_edges()),
        "umbral": umbral,
        "metodo_inversa": metodo_inversa,
    }
    return Z, A, L, multiplicadores, metricas, aristas, resumen

# ── Modelo nuevo 2: Riesgo / Markov Switching ────────────────────────────────
@st.cache_data
def cargar_dataset_markov(nombre_archivo: str = "dataset_final_modelo_markov_switching.csv") -> pd.DataFrame:
    path = DATOS_DIR / nombre_archivo
    df = pd.read_csv(path)
    df["Periodo"] = pd.to_datetime(df["Periodo"], errors="coerce")
    for c in ["TRM", "Tasa", "Flujos_Financieros", "TRM_Var", "Tasa_Var", "Volatilidad_TRM", "Volatilidad_Tasa", "Score_Riesgo", "Riesgo_Label"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Periodo").reset_index(drop=True)
    df["Alerta"] = df["Riesgo_Label"].map({0: "ALERTA VERDE", 1: "ALERTA AMARILLA", 2: "ALERTA ROJA"}).fillna("SIN CLASIFICAR")
    return df

@st.cache_data
def calcular_resumen_markov():
    df = cargar_dataset_markov()
    labels = df["Riesgo_Label"].dropna().astype(int).values
    estados = sorted(pd.Series(labels).unique().tolist()) if len(labels) else [0, 1, 2]
    k = max(estados) + 1 if estados else 3
    P = np.zeros((k, k))
    for t in range(1, len(labels)):
        P[labels[t-1], labels[t]] += 1
    rs = P.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1
    P = P / rs
    P_df = pd.DataFrame(P, index=[f"R{i}" for i in range(k)], columns=[f"R{i}" for i in range(k)])
    ultimo = int(labels[-1]) if len(labels) else 0
    prob_roja_t1 = float(P[ultimo, 2] * 100) if P.shape[1] > 2 else 0
    prob_roja_t2 = float(sum(P[ultimo, j] * P[j, 2] for j in range(P.shape[0])) * 100) if P.shape[1] > 2 else 0
    resumen = {
        "ultimo_periodo": df["Periodo"].max(),
        "ultimo_regimen": ultimo,
        "prob_roja_t1": prob_roja_t1,
        "prob_roja_t2": prob_roja_t2,
        "observaciones": int(len(df)),
    }
    return df, P_df, resumen
