"""
app.py — Dashboard principal MVES-CO
Sistema de Alerta Temprana: Riesgo de Contagio Financiero Intersectorial en Colombia
Maestría en Analítica de Datos · Universidad Central · 2026
Autores: Katy Pacheco Manchego · Julian Andres Firacative Varon
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import json
from pathlib import Path
import networkx as nx
from mves_data import (
    MACROS, MACROS_LIST, COLORES_ESTADO, NOMBRES_ESTADO,
    COLORES_REGIMEN, NOMBRES_REGIMEN, PESOS_PIB, EXCLUIR_COVID,
    cargar_panel, cargar_leontief, calcular_icds_star,
    calcular_serie_agregada, calcular_transicion, simular_choque, ejecutar_modelo_leontief_nuevo, calcular_resumen_markov,
)

def hex_to_rgba(hex_color, alpha=1.0):
    """Convierte #RRGGBB a rgba(r,g,b,a) para Plotly."""
    if not isinstance(hex_color, str):
        return f"rgba(136,136,136,{alpha})"
    h = hex_color.strip().lstrip('#')
    if len(h) != 6:
        return hex_color
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Configuración de página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="MVES-CO · Alerta Temprana",
    page_icon="🔔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personalizado ────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #1F4E79; }
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #D6E4F0 !important; }
.metric-card {
    background: white; border: 1px solid #e1e8f0; border-radius: 10px;
    padding: 14px 18px; text-align: center;
}
.metric-card .val { font-size: 28px; font-weight: 700; color: #1F4E79; }
.metric-card .lbl { font-size: 12px; color: #6b7a8d; margin-top: 2px; }
.badge-s1 { background:#EAF3DE; color:#27500A; padding:2px 9px; border-radius:20px; font-weight:600; font-size:12px; }
.badge-s2 { background:#E1F5EE; color:#085041; padding:2px 9px; border-radius:20px; font-weight:600; font-size:12px; }
.badge-s3 { background:#E6F1FB; color:#0C447C; padding:2px 9px; border-radius:20px; font-weight:600; font-size:12px; }
.badge-s4 { background:#FAEEDA; color:#633806; padding:2px 9px; border-radius:20px; font-weight:600; font-size:12px; }
.badge-s5 { background:#FCEBEB; color:#791F1F; padding:2px 9px; border-radius:20px; font-weight:600; font-size:12px; }
.alerta-BAJO  { background:#E2EFDA; color:#27500A; border:1px solid #A9C985; padding:4px 12px; border-radius:6px; font-weight:600; }
.alerta-MEDIO { background:#FAEEDA; color:#633806; border:1px solid #D4A04A; padding:4px 12px; border-radius:6px; font-weight:600; }
.alerta-ALTO  { background:#FCEBEB; color:#791F1F; border:1px solid #E07070; padding:4px 12px; border-radius:6px; font-weight:600; }
div[data-testid="stHorizontalBlock"] > div { background: white; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Cargar datos ─────────────────────────────────────────────────────────────
@st.cache_data
def get_all_data():
    panel = cargar_panel()
    L, A = cargar_leontief()
    df_star = calcular_icds_star(panel, A)
    serie_agg = calcular_serie_agregada(df_star)
    P = calcular_transicion(serie_agg["regimen"].values)
    return panel, L, A, df_star, serie_agg, P

panel, L, A, df_star, serie_agg, P_mat = get_all_data()

ult_mes = df_star.dropna(subset=["icds_star"])["fecha"].max()
ult = df_star[df_star["fecha"] == ult_mes].set_index("macrosector_id")

ultimo_r = int(serie_agg["regimen"].iloc[-1])
alerta_1 = float(P_mat[ultimo_r, 2]) * 100
alerta_2 = float(sum(P_mat[ultimo_r, j] * P_mat[j, 2] for j in range(3))) * 100
nivel_alerta = "ALTO" if alerta_1 > 40 else "MEDIO" if alerta_1 > 20 else "BAJO"

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔔 MVES-CO")
    st.markdown("**Sistema de Alerta Temprana**")
    st.markdown("Contagio Financiero Intersectorial · Colombia")
    st.markdown("---")
    pagina = st.radio("Navegación", [
        "📊 Resumen ejecutivo",
        "📈 Sectores — ICDS / ICDS*",
        "⏱ Evolución temporal",
        "🔄 Regímenes Markov",
        "🔗 Contagio MIP",
        "💥 Simulador de choque",
        "📖 Metodología",
    ])
    st.markdown("---")
    st.markdown(f"**Último mes:** `{ult_mes}`")
    st.markdown(f"**Régimen actual:** {NOMBRES_REGIMEN[ultimo_r]}")
    nivel_color = {"BAJO": "🟢", "MEDIO": "🟡", "ALTO": "🔴"}
    st.markdown(f"**Alerta:** {nivel_color[nivel_alerta]} {nivel_alerta}")
    st.markdown("---")
    st.caption("Katy Pacheco Manchego\nJulian Andres Firacative Varon\nUniversidad Central · 2026")

# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ═══════════════════════════════════════════════════════════════════════════
if pagina == "📊 Resumen ejecutivo":
    st.title("📊 Resumen Ejecutivo")
    st.markdown(f"**Sistema de Alerta Temprana — Contagio Financiero Intersectorial Colombia · {ult_mes}**")

    # Métricas principales
    icds_agg_ult = round(float(serie_agg["icds_agg"].iloc[-1]), 3)
    s45 = ult[ult["estado_star"].isin(["S4", "S5"])]
    contagio = ult[ult["ajuste_mip"] > 0.015]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ICDS* Sistémico", f"{icds_agg_ult:.3f}")
    c2.metric("Régimen actual", NOMBRES_REGIMEN[ultimo_r])
    c3.metric("Nivel de alerta", nivel_alerta,
              delta="P(contracción t+1)" + f" {alerta_1:.1f}%",
              delta_color="inverse" if alerta_1 > 20 else "normal")
    c4.metric("Sectores en riesgo (S4-S5)", f"{len(s45)} / 12")
    c5.metric("Con contagio MIP detectado", f"{len(contagio)} / 12")

    st.divider()
    col_izq, col_der = st.columns([1.1, 0.9])

    with col_izq:
        st.subheader("Estado actual por macrosector")
        sorted_m = ult.sort_values("icds_star", ascending=False)
        for m, row in sorted_m.iterrows():
            est = row.get("estado_star", "S4") or "S4"
            color = COLORES_ESTADO.get(est, "#888")
            nombre_est = NOMBRES_ESTADO.get(est, est)
            ajuste = row.get("ajuste_mip", 0) or 0
            icds_v = row.get("icds_star", 0) or 0
            alerta_c = "⚠️" if ajuste > 0.015 else ""
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:8px;
                padding:7px 10px;border-bottom:1px solid #e1e8f0;font-size:13px">
                <span class="badge-{est.lower()}">{est}</span>
                <span style="flex:1">{MACROS[m]}</span>
                <div style="width:100px;background:#f0f4f8;border-radius:3px;height:6px;overflow:hidden">
                  <div style="width:{int(icds_v*100)}%;height:6px;background:{color};border-radius:3px"></div>
                </div>
                <span style="min-width:45px;text-align:right;font-weight:500">{icds_v:.3f}</span>
                <span style="font-size:11px;color:#E24B4A">{alerta_c}</span>
                </div>""", unsafe_allow_html=True)

    with col_der:
        st.subheader("Distribución de estados (ICDS*)")
        dist = ult["estado_star"].value_counts().reindex(["S1","S2","S3","S4","S5"]).fillna(0)
        fig_pie = go.Figure(go.Pie(
            labels=[f"{k} {NOMBRES_ESTADO[k]}" for k in dist.index],
            values=dist.values,
            marker_colors=[COLORES_ESTADO[k] for k in dist.index],
            hole=0.45, textinfo="value+percent",
            textfont_size=12,
        ))
        fig_pie.update_layout(height=260, margin=dict(t=10, b=10, l=10, r=10),
                               legend=dict(font_size=11))
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Probabilidades de alerta (Markov)")
        fig_alerta = go.Figure(go.Bar(
            x=["P(R3 en t+1)", "P(R3 en t+2)"],
            y=[alerta_1, alerta_2],
            marker_color=["#E24B4A" if alerta_1 > 40 else "#BA7517" if alerta_1 > 20 else "#1D9E75",
                          "#E24B4A" if alerta_2 > 40 else "#BA7517" if alerta_2 > 20 else "#1D9E75"],
            text=[f"{alerta_1:.1f}%", f"{alerta_2:.1f}%"],
            textposition="outside",
        ))
        fig_alerta.add_hline(y=40, line_dash="dash", line_color="#E24B4A",
                              annotation_text="Umbral ALTO", annotation_font_size=10)
        fig_alerta.add_hline(y=20, line_dash="dash", line_color="#BA7517",
                              annotation_text="Umbral MEDIO", annotation_font_size=10)
        fig_alerta.update_layout(height=180, margin=dict(t=10,b=10,l=10,r=10),
                                  yaxis=dict(range=[0, max(alerta_2 + 15, 60)],
                                             title="Probabilidad (%)"))
        st.plotly_chart(fig_alerta, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — SECTORES ICDS / ICDS*
# ═══════════════════════════════════════════════════════════════════════════
elif pagina == "📈 Sectores — ICDS / ICDS*":
    st.title("📈 Clasificación Sectorial — ICDS vs ICDS*")
    st.markdown(f"**{ult_mes}** · Comparación del índice original vs ajustado por contagio MIP")

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        st.subheader("Tabla comparativa")
        sorted_df = ult.sort_values("icds_star", ascending=False).reset_index()
        tabla = []
        for _, row in sorted_df.iterrows():
            m = row["macrosector_id"]
            tabla.append({
                "Macrosector": MACROS[m],
                "ICDS": round(float(row.get("icds", 0) or 0), 4),
                "ICDS*": round(float(row.get("icds_star", 0) or 0), 4),
                "Ajuste MIP": round(float(row.get("ajuste_mip", 0) or 0), 4),
                "Estado ICDS": row.get("estado", "") or "",
                "Estado ICDS*": row.get("estado_star", "") or "",
            })
        df_tabla = pd.DataFrame(tabla)

        def color_estado(val):
            c = {"S1": "background-color:#EAF3DE", "S2": "background-color:#E1F5EE",
                 "S3": "background-color:#E6F1FB", "S4": "background-color:#FAEEDA",
                 "S5": "background-color:#FCEBEB"}.get(val, "")
            return c

        st.dataframe(
            df_tabla.style
            .applymap(color_estado, subset=["Estado ICDS", "Estado ICDS*"])
            .background_gradient(subset=["ICDS", "ICDS*"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({"ICDS": "{:.4f}", "ICDS*": "{:.4f}", "Ajuste MIP": "{:.4f}"}),
            use_container_width=True, height=450,
        )

    with col2:
        st.subheader("Ranking por ICDS*")
        fig_rank = go.Figure()
        sorted_df2 = ult.sort_values("icds_star")
        labels = [MACROS[m][:20] for m in sorted_df2.index]
        colors = [COLORES_ESTADO.get(str(sorted_df2.loc[m, "estado_star"]), "#888")
                  for m in sorted_df2.index]
        fig_rank.add_trace(go.Bar(
            y=labels, x=sorted_df2["icds"].values,
            orientation="h", name="ICDS",
            marker_color="rgba(55,138,221,.4)",
            marker_line_color="#378ADD", marker_line_width=1,
        ))
        fig_rank.add_trace(go.Bar(
            y=labels, x=sorted_df2["icds_star"].values,
            orientation="h", name="ICDS*",
            marker_color=[hex_to_rgba(c, 0.60) for c in colors],
            marker_line_color=colors, marker_line_width=1.5,
        ))
        for x in [0.40, 0.60]:
            fig_rank.add_vline(x=x, line_dash="dash",
                                line_color="gray", opacity=0.4, line_width=1)
        fig_rank.update_layout(
            barmode="overlay", height=430,
            margin=dict(t=10, b=10, l=5, r=5),
            legend=dict(orientation="h", y=1.05, font_size=11),
            xaxis=dict(range=[0, 1], title="ICDS"),
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    st.divider()
    st.subheader("Heatmap ICDS* — todos los sectores y períodos")
    fechas_sample = [f for f in sorted(df_star["fecha"].unique()) if f >= "2019-01"]
    fechas_tick = fechas_sample[::3]
    pivot = df_star.pivot_table(index="macrosector_id", columns="fecha",
                                 values="icds_star", aggfunc="mean")
    pivot = pivot[[c for c in pivot.columns if c in fechas_sample]]
    pivot.index = [MACROS[m][:22] for m in pivot.index]
    fig_heat = px.imshow(
        pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
        labels=dict(x="Mes", y="Macrosector", color="ICDS*"),
        aspect="auto", text_auto=".2f",
    )
    fig_heat.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10),
                            coloraxis_colorbar=dict(title="ICDS*", tickfont_size=10))
    fig_heat.update_xaxes(tickangle=35, tickfont_size=9,
                           tickvals=fechas_tick, ticktext=[f[:7] for f in fechas_tick])
    fig_heat.update_traces(textfont_size=8)
    st.plotly_chart(fig_heat, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — EVOLUCIÓN TEMPORAL
# ═══════════════════════════════════════════════════════════════════════════
elif pagina == "⏱ Evolución temporal":
    st.title("⏱ Evolución Temporal del ICDS")

    c1, c2, c3 = st.columns(3)
    sector_sel = c1.selectbox("Macrosector", MACROS_LIST,
                               format_func=lambda m: f"{m} — {MACROS[m]}")
    indicador_sel = c2.selectbox("Indicador", ["icds", "icds_star"],
                                  format_func=lambda x: "ICDS original" if x == "icds" else "ICDS* (ajustado MIP)")
    desde = c3.selectbox("Período desde", ["2015-01", "2018-01", "2019-01", "2022-01"])

    sub = df_star[(df_star["macrosector_id"] == sector_sel) &
                  (df_star["fecha"] >= desde)].sort_values("fecha")
    vals = sub[indicador_sel].values
    fechas_s = sub["fecha"].values
    estados = (sub["estado"] if indicador_sel == "icds" else sub["estado_star"]).values

    fig_evol = go.Figure()
    # Zonas de fondo
    for ymin, ymax, color, nombre in [
        (0.80, 1.00, "#639922", "S1"), (0.60, 0.80, "#1D9E75", "S2"),
        (0.40, 0.60, "#378ADD", "S3"), (0.20, 0.40, "#BA7517", "S4"),
        (0.00, 0.20, "#E24B4A", "S5"),
    ]:
        fig_evol.add_hrect(y0=ymin, y1=ymax, fillcolor=color, opacity=0.06,
                            line_width=0, annotation_text=nombre,
                            annotation_position="right", annotation_font_size=10)
    # Líneas umbral
    for y in [0.20, 0.40, 0.60, 0.80]:
        fig_evol.add_hline(y=y, line_dash="dot", line_color="gray", opacity=0.4, line_width=0.8)
    # COVID
    fig_evol.add_vrect(x0="2020-03", x1="2021-12", fillcolor="#E24B4A",
                        opacity=0.08, line_width=0, annotation_text="COVID-19",
                        annotation_position="top left", annotation_font_size=10)
    # Serie principal
    fig_evol.add_trace(go.Scatter(
        x=fechas_s, y=vals, mode="lines",
        name="ICDS*" if indicador_sel == "icds_star" else "ICDS",
        line=dict(color="#1F4E79", width=2),
        fill="tozeroy", fillcolor="rgba(55,138,221,0.07)",
    ))
    fig_evol.update_layout(
        title=f"{MACROS[sector_sel]} — {'ICDS*' if indicador_sel=='icds_star' else 'ICDS'} mensual",
        yaxis=dict(range=[0, 1], title="Índice"),
        xaxis=dict(title="Mes"),
        height=340, margin=dict(t=40, b=20, l=10, r=80),
        legend=dict(font_size=11),
    )
    st.plotly_chart(fig_evol, use_container_width=True)

    st.divider()
    st.subheader("Comparar múltiples sectores")
    sectores_comp = st.multiselect(
        "Selecciona sectores", MACROS_LIST,
        default=["M03", "M05", "M08"],
        format_func=lambda m: f"{m} — {MACROS[m]}",
    )
    if sectores_comp:
        fig_comp = go.Figure()
        colores_comp = px.colors.qualitative.Set2
        for i, m in enumerate(sectores_comp):
            sub_c = df_star[(df_star["macrosector_id"] == m) &
                             (df_star["fecha"] >= desde)].sort_values("fecha")
            fig_comp.add_trace(go.Scatter(
                x=sub_c["fecha"], y=sub_c["icds_star"],
                name=MACROS[m][:25], mode="lines",
                line=dict(color=colores_comp[i % len(colores_comp)], width=1.8),
            ))
        for y in [0.20, 0.40, 0.60, 0.80]:
            fig_comp.add_hline(y=y, line_dash="dot", line_color="gray", opacity=0.3, line_width=0.8)
        fig_comp.add_vrect(x0="2020-03", x1="2021-12", fillcolor="#E24B4A",
                            opacity=0.07, line_width=0)
        fig_comp.update_layout(
            title="ICDS* comparativo por sector seleccionado",
            yaxis=dict(range=[0, 1], title="ICDS*"),
            height=320, margin=dict(t=40, b=20, l=10, r=10),
            legend=dict(font_size=10, orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — REGÍMENES MARKOV
# ═══════════════════════════════════════════════════════════════════════════
elif pagina == "🔄 Regímenes Markov":
    st.title("🔄 Detección de Regímenes — Markov-Switching")
    st.markdown("Regímenes no observados inferidos sobre el ICDS* agregado ponderado por PIB")

    # Serie con regímenes
    fig_reg = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             row_heights=[0.65, 0.35],
                             vertical_spacing=0.05)
    # Zonas de régimen
    for r in range(3):
        mask = serie_agg["regimen"] == r
        fechas_r = serie_agg["fecha"][mask].values
        if len(fechas_r) > 0:
            fig_reg.add_trace(go.Scatter(
                x=serie_agg["fecha"], y=np.where(mask, serie_agg["icds_agg"], np.nan),
                mode="lines", line=dict(color=COLORES_REGIMEN[r], width=0),
                fill="tozeroy", fillcolor=hex_to_rgba(COLORES_REGIMEN[r], 0.12),
                name=NOMBRES_REGIMEN[r], showlegend=True,
            ), row=1, col=1)
    fig_reg.add_trace(go.Scatter(
        x=serie_agg["fecha"], y=serie_agg["icds_agg"],
        mode="lines", line=dict(color="#1F4E79", width=2),
        name="ICDS* agregado", showlegend=True,
    ), row=1, col=1)
    for y in [0.40, 0.60]:
        fig_reg.add_hline(y=y, line_dash="dot", line_color="gray",
                           opacity=0.4, line_width=0.8, row=1, col=1)
    # Panel de probabilidades (barras de régimen)
    colores_bar = [COLORES_REGIMEN[r] for r in serie_agg["regimen"]]
    fig_reg.add_trace(go.Bar(
        x=serie_agg["fecha"], y=[1] * len(serie_agg),
        marker_color=colores_bar, marker_line_width=0,
        name="Régimen dominante", showlegend=False,
    ), row=2, col=1)
    fig_reg.update_layout(
        height=480, margin=dict(t=20, b=20, l=10, r=10),
        legend=dict(orientation="h", y=1.05, font_size=11),
        yaxis=dict(range=[0, 1], title="ICDS*"),
        yaxis2=dict(showticklabels=False, title="Régimen"),
    )
    st.plotly_chart(fig_reg, use_container_width=True)

    st.divider()
    col_p, col_dur = st.columns(2)

    with col_p:
        st.subheader("Matriz de transición P")
        st.markdown("P_ij = probabilidad de pasar del estado i al estado j en el próximo mes")
        df_P = pd.DataFrame(
            P_mat * 100,
            index=[NOMBRES_REGIMEN[i] for i in range(3)],
            columns=[NOMBRES_REGIMEN[j] for j in range(3)],
        ).round(1)
        st.dataframe(
            df_P.style
            .background_gradient(cmap="RdYlGn", vmin=0, vmax=100)
            .format("{:.1f}%"),
            use_container_width=True,
        )
        st.caption("Estimación empírica basada en la secuencia histórica de regímenes detectados.")

    with col_dur:
        st.subheader("Duración media y alerta")
        for r in range(3):
            dur = 1 / (1 - P_mat[r, r]) if P_mat[r, r] < 1 else float("inf")
            p_cont = P_mat[r, 2] * 100
            col_r = COLORES_REGIMEN[r]
            st.markdown(
                f"""<div style="padding:10px 14px;border-left:4px solid {col_r};
                    background:{hex_to_rgba(col_r, 0.10)};border-radius:6px;margin-bottom:8px">
                    <b>{NOMBRES_REGIMEN[r]}</b><br>
                    <span style="font-size:12px">
                    Duración media: <b>{dur:.1f} meses</b> · 
                    Persistencia: <b>{P_mat[r,r]*100:.1f}%</b> · 
                    P(→ Contracción): <b>{p_cont:.1f}%</b>
                    </span></div>""",
                unsafe_allow_html=True,
            )
        st.markdown("---")
        st.markdown(f"**Régimen actual:** {NOMBRES_REGIMEN[ultimo_r]}")
        st.markdown(f"**P(contracción t+1):** {alerta_1:.1f}%")
        st.markdown(f"**P(contracción t+2):** {alerta_2:.1f}%")
        nivel_color = {"BAJO": "🟢", "MEDIO": "🟡", "ALTO": "🔴"}
        st.markdown(f"**Nivel de alerta:** {nivel_color[nivel_alerta]} **{nivel_alerta}**")


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — CONTAGIO MIP
# ═══════════════════════════════════════════════════════════════════════════


    st.divider()

    # ═══════════════════════════════════════════════════════════════════════════
    # BLOQUE INTEGRADO — MODELO DE RIESGO MARKOV SWITCHING
    # ═══════════════════════════════════════════════════════════════════════════
    st.title("📉 Modelo de Riesgo — Markov Switching")
    st.markdown("Modelo calculado desde `datos/dataset_final_modelo_markov_switching.csv`.")

    df_mk, P_df, resumen_mk = calcular_resumen_markov()
    ult_reg = resumen_mk["ultimo_regimen"]
    nombre_reg = {0: "R0 Verde", 1: "R1 Amarillo", 2: "R2 Rojo"}.get(ult_reg, f"R{ult_reg}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observaciones", resumen_mk["observaciones"])
    c2.metric("Último periodo", resumen_mk["ultimo_periodo"].strftime("%Y-%m") if pd.notna(resumen_mk["ultimo_periodo"]) else "N/A")
    c3.metric("Régimen actual", nombre_reg)
    c4.metric("Prob. alerta roja t+1", f'{resumen_mk["prob_roja_t1"]:.1f}%')


   ### with col2:
      ###  st.subheader("Matriz de transición")
        ###fig_h = px.imshow(P_df, text_auto=".2%", aspect="auto", color_continuous_scale="Blues")
        ###fig_h.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
        ###st.plotly_chart(fig_h, use_container_width=True)


elif pagina == "🔗 Contagio MIP":
    st.title("Contagio Intersectorial — Matriz Insumo-Producto")
    st.markdown("Vista organizada del modelo Leontief, dependencias productivas y contagio sectorial.")

    # ═══════════════════════════════════════════════════════════════════════
    # ESTILOS DE LA PÁGINA
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("""
    <style>
    .mip-hero-clean {
        background: #FFFFFF;
        padding: 10px 0 20px 0;
        margin-bottom: 10px;
        border-bottom: 1px solid #E5E7EB;
    }
    .mip-hero-clean h2 {
        margin: 0 0 8px 0;
        font-size: 34px;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.4px;
    }
    .mip-hero-clean p {
        margin: 0;
        font-size: 17px;
        color: #475569;
        line-height: 1.7;
        max-width: 980px;
    }
    .mip-card {
        background: white;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,.06);
        border: 1px solid #E5E7EB;
        margin-bottom: 14px;
    }
    .mip-kpi {
        background: white;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,.06);
        border-left: 6px solid #1F4E79;
        min-height: 92px;
    }
    .mip-kpi-label {
        color:#64748B;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .mip-kpi-value {
        color:#0F172A;
        font-size: 26px;
        font-weight: 850;
        line-height: 1.1;
    }
    .mip-note {
        background:#F8FAFC;
        border-left:5px solid #2E75B6;
        padding:14px 16px;
        border-radius:12px;
        color:#334155;
        font-size:14px;
        margin: 12px 0 16px 0;
    }
    .mip-alert {
        background:#FFF7ED;
        border-left:5px solid #EA580C;
        padding:14px 16px;
        border-radius:12px;
        color:#7C2D12;
        font-size:14px;
        margin: 12px 0 16px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mip-hero-clean">
        <h2>Modelo Leontief + Red de Contagio Sectorial</h2>
        <p>
        Identifique sectores críticos, vínculos productivos relevantes, efectos multiplicadores
        y mecanismos de propagación de choques económicos mediante la Matriz Insumo-Producto de Colombia.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # CÁLCULOS BASE
    # ═══════════════════════════════════════════════════════════════════════
    # Matriz A sin diagonal: uso to_numpy(copy=True) para evitar errores de arrays de solo lectura en Streamlit Cloud
    A_off_np = A.fillna(0).astype(float).to_numpy(copy=True)
    np.fill_diagonal(A_off_np, 0)

    A_off = pd.DataFrame(
        A_off_np,
        index=A.index,
        columns=A.columns
    )

    mult_df = pd.DataFrame({
        "ID": MACROS_LIST,
        "Sector": [MACROS[m] for m in MACROS_LIST],
        "Multiplicador": [float(L.loc[m, m]) for m in MACROS_LIST],
    }).sort_values("Multiplicador", ascending=False)

    pares = []
    for i in MACROS_LIST:
        for j in MACROS_LIST:
            v = float(A_off.loc[i, j])
            if v > 0:
                pares.append((v, i, j))
    pares.sort(reverse=True)

    df_pares = pd.DataFrame(
        [(MACROS[i], MACROS[j], v) for v, i, j in pares[:15]],
        columns=["Sector origen/proveedor", "Sector destino/comprador", "Coeficiente técnico"]
    )

    total_sectores = len(MACROS_LIST)
    total_vinculos = len([x for x in pares if x[0] > 0.02])
    mult_max = mult_df["Multiplicador"].max()
    sector_mult_max = mult_df.iloc[0]["Sector"]
    vinculo_mas_fuerte = df_pares.iloc[0] if len(df_pares) else None

    # Indicador simple de criticidad: multiplicador + suma de vínculos de salida + suma de vínculos de entrada
    criticidad = []
    for m in MACROS_LIST:
        salida = float(A_off.loc[m, :].sum())
        entrada = float(A_off.loc[:, m].sum())
        mult = float(L.loc[m, m])
        score = (mult / max(mult_df["Multiplicador"])) * 0.50 + salida * 0.25 + entrada * 0.25
        criticidad.append({
            "ID": m,
            "Sector": MACROS[m],
            "Criticidad": score,
            "Multiplicador": mult,
            "Vínculos salida": salida,
            "Vínculos entrada": entrada,
        })

    df_criticidad = pd.DataFrame(criticidad).sort_values("Criticidad", ascending=False)
    df_criticidad["Nivel"] = pd.cut(
        df_criticidad["Criticidad"],
        bins=[-np.inf, df_criticidad["Criticidad"].quantile(.50), df_criticidad["Criticidad"].quantile(.80), np.inf],
        labels=["Medio", "Alto", "Crítico"]
    )

    # ═══════════════════════════════════════════════════════════════════════
    # KPIS SUPERIORES
    # ═══════════════════════════════════════════════════════════════════════
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="mip-kpi">
            <div class="mip-kpi-label">Sectores analizados</div>
            <div class="mip-kpi-value">{total_sectores}</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="mip-kpi">
            <div class="mip-kpi-label">Vínculos productivos relevantes</div>
            <div class="mip-kpi-value">{total_vinculos}</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="mip-kpi">
            <div class="mip-kpi-label">Multiplicador máximo</div>
            <div class="mip-kpi-value">{mult_max:.2f}x</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="mip-kpi">
            <div class="mip-kpi-label">Sector más crítico</div>
            <div class="mip-kpi-value" style="font-size:18px;">{df_criticidad.iloc[0]['Sector'][:28]}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="mip-note">
    <b>Lectura rápida:</b> esta página responde tres preguntas: <b>qué sectores son críticos</b>,
    <b>qué sector contagia a cuál</b> y <b>qué tan fuerte se amplifica un choque</b> mediante la red productiva.
    </div>
    """, unsafe_allow_html=True)

    tab_resumen, tab_red, tab_contagio, tab_mult, tab_datos = st.tabs([
        "1️⃣ Resumen ejecutivo",
        "2️⃣ Red productiva",
        "3️⃣ Contagio directo",
        "4️⃣ Multiplicadores",
        "5️⃣ Datos"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 — RESUMEN EJECUTIVO
    # ═══════════════════════════════════════════════════════════════════════
    with tab_resumen:
        st.subheader("🔴 Sectores con mayor criticidad")

        top_crit = df_criticidad.head(8).sort_values("Criticidad", ascending=True)

        fig_crit = px.bar(
            top_crit,
            x="Criticidad",
            y="Sector",
            orientation="h",
            text="Criticidad",
            color="Nivel",
            color_discrete_map={
                "Crítico": "#E24B4A",
                "Alto": "#D9A514",
                "Medio": "#6BAED6"
            }
        )
        fig_crit.update_traces(texttemplate="%{text:.2f}", textposition="outside", marker_line_width=0)
        fig_crit.update_layout(
            height=390,
            template="plotly_white",
            xaxis_title="Índice de criticidad compuesto",
            yaxis_title="",
            margin=dict(l=10, r=35, t=10, b=35),
            legend=dict(orientation="h", y=1.12)
        )
        st.plotly_chart(fig_crit, use_container_width=True)

        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown("""
            <div class="mip-card">
            <b>¿Cómo se interpreta?</b><br>
            Un sector es más crítico cuando combina: alto multiplicador de Leontief,
            alta dependencia de otros sectores y alta capacidad de transmitir choques.
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            if vinculo_mas_fuerte is not None:
                st.markdown(f"""
                <div class="mip-card">
                <b>Vínculo más fuerte detectado</b><br>
                {vinculo_mas_fuerte['Sector origen/proveedor']} → {vinculo_mas_fuerte['Sector destino/comprador']}<br>
                <b>Coeficiente:</b> {vinculo_mas_fuerte['Coeficiente técnico']:.4f}
                </div>
                """, unsafe_allow_html=True)

        st.success(
            f"El sector con mayor multiplicador es {sector_mult_max}. "
            "Esto sugiere que un choque en sectores centrales puede amplificarse y propagarse hacia otras actividades conectadas."
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 — RED PRODUCTIVA
    # ═══════════════════════════════════════════════════════════════════════
    with tab_red:
        st.subheader("🌐 Red de dependencias productivas")

        col_ctrl, col_plot = st.columns([0.25, 0.75])

        with col_ctrl:
            umbral = st.slider(
                "Umbral mínimo del vínculo",
                0.01,
                0.15,
                0.04,
                0.01,
                help="Sube el umbral para ver solo relaciones más fuertes."
            )
            st.markdown("""
            <div class="mip-note">
            Los nodos son sectores. Las líneas representan vínculos productivos relevantes.
            El tamaño del nodo aumenta según su multiplicador.
            </div>
            """, unsafe_allow_html=True)

        with col_plot:
            # Copia numérica segura para Streamlit Cloud
            A_np = A.fillna(0).astype(float).to_numpy(copy=True)
            np.fill_diagonal(A_np, 0)

            G = nx.DiGraph()
            for n in MACROS_LIST:
                G.add_node(n)
            for i, o in enumerate(MACROS_LIST):
                for j, d in enumerate(MACROS_LIST):
                    if A_np[i, j] > umbral:
                        G.add_edge(o, d, weight=A_np[i, j])

            pos = nx.circular_layout(G, scale=2)
            colores_nodo = px.colors.qualitative.Set3[:12]
            mult_diag = {m: float(L.loc[m, m]) for m in MACROS_LIST}

            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            node_size = [18 + 32 * mult_diag.get(n, 1) for n in G.nodes()]
            node_text = [f"{n}<br>{MACROS[n]}<br>Multiplicador: {mult_diag[n]:.3f}" for n in G.nodes()]

            fig_net = go.Figure()
            fig_net.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="rgba(100,100,100,0.25)", width=1.2),
                hoverinfo="none",
            ))
            fig_net.add_trace(go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_size,
                    color=colores_nodo[:len(node_x)],
                    line=dict(width=1.5, color="white")
                ),
                text=list(G.nodes()),
                textposition="middle center",
                textfont=dict(size=9, color="black", family="Arial Black"),
                hovertext=node_text,
                hoverinfo="text",
            ))
            fig_net.update_layout(
                showlegend=False,
                height=470,
                template="plotly_white",
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            )
            st.plotly_chart(fig_net, use_container_width=True)

        st.subheader("Top vínculos intersectoriales")
        st.dataframe(
            df_pares.head(10).style.format({"Coeficiente técnico": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 — CONTAGIO DIRECTO
    # ═══════════════════════════════════════════════════════════════════════
    with tab_contagio:
        st.subheader("🦠 ¿Qué sector contagia a cuál?")

        col_sim, col_res = st.columns([0.30, 0.70])

        with col_sim:
            sector_origen = st.selectbox(
                "Sector que recibe el choque inicial",
                MACROS_LIST,
                format_func=lambda m: f"{m} — {MACROS[m]}"
            )
            shock = st.slider(
                "Intensidad del choque inicial (%)",
                min_value=5,
                max_value=50,
                value=20,
                step=5
            )
            top_n = st.slider("Número de sectores destino", 3, 10, 6, 1)

            st.markdown("""
            <div class="mip-note">
            El impacto estimado se calcula como:<br>
            <b>Coeficiente técnico × intensidad del choque</b>.
            </div>
            """, unsafe_allow_html=True)

        with col_res:
            impactos = []
            for destino in MACROS_LIST:
                if destino == sector_origen:
                    continue
                coef = float(A_off.loc[sector_origen, destino])
                if coef > 0:
                    impactos.append({
                        "Sector destino": MACROS[destino],
                        "ID destino": destino,
                        "Coeficiente": coef,
                        "Impacto estimado (%)": coef * shock
                    })

            df_impactos = pd.DataFrame(impactos).sort_values("Impacto estimado (%)", ascending=False)

            if df_impactos.empty:
                st.info("Con el sector seleccionado no se detectan vínculos directos relevantes en la matriz.")
            else:
                df_plot = df_impactos.head(top_n).sort_values("Impacto estimado (%)", ascending=True)

                fig_imp = px.bar(
                    df_plot,
                    x="Impacto estimado (%)",
                    y="Sector destino",
                    orientation="h",
                    text="Impacto estimado (%)",
                    color="Impacto estimado (%)",
                    color_continuous_scale="Reds"
                )
                fig_imp.update_traces(texttemplate="%{text:.2f}%", textposition="outside", marker_line_width=0)
                fig_imp.update_layout(
                    height=390,
                    template="plotly_white",
                    coloraxis_showscale=False,
                    xaxis_title="Impacto estimado sobre sector destino",
                    yaxis_title="",
                    margin=dict(l=10, r=40, t=10, b=35)
                )
                st.plotly_chart(fig_imp, use_container_width=True)

                mayor = df_impactos.iloc[0]
                st.markdown(f"""
                <div class="mip-alert">
                <b>Resultado:</b> si <b>{MACROS[sector_origen]}</b> recibe un choque del <b>{shock}%</b>,
                el sector más afectado directamente sería <b>{mayor['Sector destino']}</b>,
                con impacto estimado de <b>{mayor['Impacto estimado (%)']:.2f}%</b>.
                </div>
                """, unsafe_allow_html=True)

        st.subheader("Mapa simple de contagio directo")
        if not df_impactos.empty:
            df_edges = df_impactos.head(12).copy()
            df_edges["Origen"] = MACROS[sector_origen]
            fig_edges = px.scatter(
                df_edges,
                x="Origen",
                y="Sector destino",
                size="Coeficiente",
                color="Impacto estimado (%)",
                text="Impacto estimado (%)",
                size_max=48,
                color_continuous_scale="Reds"
            )
            fig_edges.update_traces(texttemplate="%{text:.2f}%", textposition="top center")
            fig_edges.update_layout(
                height=430,
                template="plotly_white",
                coloraxis_showscale=True,
                xaxis_title="Sector origen del choque",
                yaxis_title="Sector destino del contagio",
                margin=dict(l=10, r=20, t=10, b=35)
            )
            st.plotly_chart(fig_edges, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 — MULTIPLICADORES
    # ═══════════════════════════════════════════════════════════════════════
    with tab_mult:
        st.subheader("📈 Multiplicadores de Leontief")

        fig_mult = go.Figure(go.Bar(
            y=mult_df["Sector"].str[:30],
            x=mult_df["Multiplicador"],
            orientation="h",
            marker_color="#2E75B6",
            marker_opacity=0.85,
            text=mult_df["Multiplicador"].round(3)
        ))
        fig_mult.update_traces(texttemplate="%{text:.3f}x", textposition="outside")
        fig_mult.add_vline(x=1, line_dash="dash", line_color="gray", opacity=0.5)
        fig_mult.update_layout(
            height=520,
            template="plotly_white",
            xaxis_title="Multiplicador diagonal de Leontief (Lᵢᵢ)",
            yaxis_title="",
            margin=dict(t=10, b=35, l=10, r=35)
        )
        st.plotly_chart(fig_mult, use_container_width=True)

        st.markdown("""
        <div class="mip-note">
        <b>Interpretación:</b> un multiplicador superior a 1 indica que el efecto total de un choque puede ser mayor
        que el impacto inicial. Mientras más alto sea el multiplicador, mayor es la capacidad de amplificación del sector
        dentro de la red productiva.
        </div>
        """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 — DATOS
    # ═══════════════════════════════════════════════════════════════════════
    with tab_datos:
        st.subheader("📋 Datos usados en la página")

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown("**Top sectores por criticidad**")
            st.dataframe(
                df_criticidad[["ID", "Sector", "Criticidad", "Multiplicador", "Vínculos salida", "Vínculos entrada", "Nivel"]]
                .style.format({
                    "Criticidad": "{:.4f}",
                    "Multiplicador": "{:.4f}",
                    "Vínculos salida": "{:.4f}",
                    "Vínculos entrada": "{:.4f}"
                }),
                use_container_width=True,
                hide_index=True,
                height=360
            )

        with col_d2:
            st.markdown("**Top vínculos técnicos A**")
            st.dataframe(
                df_pares.style.format({"Coeficiente técnico": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
                height=360
            )

        csv_crit = df_criticidad.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar criticidad sectorial CSV",
            csv_crit,
            "criticidad_sectorial_mip.csv",
            "text/csv"
        )


elif pagina == "💥 Simulador de choque":
    st.title("💥 Simulador de Choque Sectorial")
    st.markdown("Simula un choque en un sector y observa cómo se propaga el contagio vía MIP")

    col1, col2 = st.columns([1, 2])
    with col1:
        sector_choque = st.selectbox(
            "Sector en crisis", MACROS_LIST,
            index=4,  # Construcción por defecto
            format_func=lambda m: f"{m} — {MACROS[m]}",
        )
        icds_crisis_val = st.slider(
            f"ICDS del sector en crisis", 0.00, 0.39, 0.10, 0.01,
            help="Valores < 0.40 = sector en Decrecimiento o Contracción",
        )
        st.metric(
            "Estado simulado",
            "S5 Contracción" if icds_crisis_val < 0.20 else "S4 Decrecimiento",
        )
        st.markdown("---")
        st.markdown("**Interpretación**")
        st.markdown(
            "El simulador aplica la fórmula:\n\n"
            "```\nICDS*_i = ICDS_i − 0.5 × Σ_j [A_ji × max(0, 0.5 − ICDS_j)]\n```\n\n"
            "donde A_ji son los coeficientes técnicos de la MIP."
        )

    with col2:
        resultado = simular_choque(df_star, A, sector_choque, icds_crisis_val)
        fig_choque = go.Figure()
        fig_choque.add_trace(go.Bar(
            y=resultado["macrosector"].str[:20],
            x=resultado["icds_base"],
            orientation="h", name="ICDS base",
            marker_color="rgba(55,138,221,0.5)",
            marker_line_color="#378ADD", marker_line_width=1,
        ))
        fig_choque.add_trace(go.Bar(
            y=resultado["macrosector"].str[:20],
            x=resultado["icds_shock"],
            orientation="h", name="ICDS post-choque",
            marker_color="rgba(226,75,74,0.5)",
            marker_line_color="#E24B4A", marker_line_width=1,
        ))
        fig_choque.add_vline(x=0.40, line_dash="dash",
                              line_color="#BA7517", opacity=0.5, line_width=1)
        fig_choque.add_vline(x=0.60, line_dash="dash",
                              line_color="#1D9E75", opacity=0.5, line_width=1)
        fig_choque.update_layout(
            barmode="overlay", height=400,
            title=f"Impacto del choque: {MACROS[sector_choque]} → ICDS={icds_crisis_val:.2f}",
            xaxis=dict(range=[0, 1], title="ICDS"),
            margin=dict(t=40, b=10, l=5, r=10),
            legend=dict(orientation="h", y=1.1, font_size=11),
        )
        st.plotly_chart(fig_choque, use_container_width=True)

    st.subheader("Tabla de impactos")
    resultado["Alerta"] = resultado["caida"].apply(
        lambda v: "⚠️ CONTAGIO" if v > 0.02 else ("🔴 EPICENTRO" if v == resultado["caida"].max() else "—")
    )
    resultado.columns = ["ID", "Sector", "ICDS base", "ICDS shock", "Caída", "Alerta"]
    st.dataframe(
        resultado[["Sector", "ICDS base", "ICDS shock", "Caída", "Alerta"]]
        .style.background_gradient(subset=["Caída"], cmap="Reds")
        .format({"ICDS base": "{:.4f}", "ICDS shock": "{:.4f}", "Caída": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA 7 — METODOLOGÍA
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# PÁGINA NUEVA — MODELO LEONTIEF CON BASE DANE LIMPIA
# ═══════════════════════════════════════════════════════════════════════════
elif pagina == "📖 Metodología":
    st.title("📖 Metodología — Modelo MVES-CO")

    st.markdown("""
    ## Marco metodológico integrado

    El **MVES-CO** (Modelo de Vulnerabilidad Económica Sectorial Colombia) integra tres capas analíticas:

    ---

    ### Capa 1 — Índice Compuesto de Desempeño Sectorial (ICDS)

    El ICDS combina cuatro dimensiones mediante normalización por percentil histórico:

    $$ICDS_i = w_1 \\cdot ISE_{norm} + w_2 \\cdot EMP_{norm} + w_3 \\cdot COSTO_{norm} + w_4 \\cdot TPM_{norm}$$

    | Dimensión | Variable | Peso | Fuente |
    |-----------|----------|------|--------|
    | ΔActividad | ISE DANE | 40% | DANE — Indicador de Seguimiento a la Economía |
    | ΔEmpleo | GEIH | 25% | DANE — Gran Encuesta Integrada de Hogares |
    | Costos | IPP / IPC | 20% | DANE — Índice de Precios al Productor |
    | Financiero | TPM | 15% | BanRep — Tasa de Política Monetaria |

    **Normalización:** Probability Integral Transform (PIT) sobre distribución histórica de cada sector,
    excluyendo COVID+rebote 2020-2021 (Reinhart & Rogoff, 2009).

    **Escala de clasificación:**

    | Estado | ICDS | Desempeño | Vulnerabilidad |
    |--------|------|-----------|----------------|
    | S1 Aceleración | ≥ 0.80 | Positivo | Baja |
    | S2 Crecimiento | 0.60 – 0.79 | Bueno | Moderada |
    | S3 Estabilidad | 0.40 – 0.59 | Aceptable | Moderada |
    | S4 Decrecimiento | 0.20 – 0.39 | Regular | Media |
    | S5 Contracción | < 0.20 | Deficiente | Alta |

    ---

    ### Capa 2 — ICDS* ajustado por Matriz Insumo-Producto

    La inversa de Leontief $L = (I - A)^{-1}$ captura los efectos en cadena de un deterioro sectorial:

    $$ICDS^*_i = ICDS_i - 0.5 \\times \\sum_j A_{ji} \\cdot \\max(0, \\ 0.5 - ICDS_j)$$

    **Fuentes MIP:** DANE — Matrices Insumo-Producto 2019 y 2021 (ponderación 40%/60%).

    **Interpretación del ajuste:** Si el sector j (proveedor de i) está deteriorado (ICDS_j < 0.5),
    transfiere riesgo a i en proporción al vínculo productivo A_ji.

    ---

    ### Capa 3 — Modelo Markov-Switching AR(1)

    $$ICDS^*_t = \\mu_{s_t} + \\phi_{s_t} \\cdot ICDS^*_{t-1} + \\sigma_{s_t} \\varepsilon_t, \\quad s_t \\sim \\text{Markov}(P)$$

    Tres regímenes no observados: **R1 Expansión** (≥0.60), **R2 Estabilidad** (0.40-0.60), **R3 Contracción** (<0.40).

    La matriz de transición P permite calcular:
    - $P(R3_{t+1} | s_t)$: señal de alerta temprana para el próximo mes
    - Duración media de cada régimen: $\\bar{d}_i = 1/(1 - p_{ii})$

    ---

    ### Referencias
    - Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series.
    - Leontief, W. (1941). The Structure of American Economy.
    - Nardo, M. et al. (2005). Tools for Composite Indicators Building. JRC European Commission.
    - OCDE (2008). Handbook on Constructing Composite Indicators.
    - Reinhart, C. & Rogoff, K. (2009). This Time Is Different. Princeton University Press.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fuentes de datos utilizadas")
        st.markdown("""
        | Fuente | Variable | Periodicidad |
        |--------|----------|-------------|
        | DANE — ISE 12 actividades | ΔActividad sectorial | Mensual |
        | DANE — GEIH Ocupados | ΔEmpleo sectorial | Mensual |
        | DANE — IPP Producción Nacional | ΔCostos productivos | Mensual |
        | BanRep — IPC Nacional | Inflación general | Mensual |
        | BanRep — TPM | Tasa política monetaria | Mensual |
        | DANE — MIP 2019 y 2021 | Coeficientes técnicos | Anual |
        """)
    with col2:
        st.subheader("Limitaciones metodológicas")
        st.markdown("""
        - **MIP estática:** Los coeficientes técnicos (2019, 2021) pueden no reflejar
          cambios estructurales post-COVID en las cadenas productivas.
        - **Normalización PIT:** Sensible al período de referencia elegido.
          Se recomienda ventana móvil en actualizaciones futuras.
        - **Agregación sectorial:** La agrupación de 68 productos CPC en 12 macrosectores
          pierde heterogeneidad interna.
        - **Pesos del ICDS:** Calibrados por juicio experto. Calibración mediante
          AHP o regresión es recomendada como trabajo futuro.
        - **Markov por umbrales:** La versión actual usa clasificación determinística.
          La estimación por MLE (statsmodels) es preferible con >100 observaciones.
        """)
