"""
app.py — Dashboard MVES-CO (versión final)
Sistema de Alerta Temprana: Contagio Financiero Intersectorial Colombia
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
from mves_data import (
    MACROS, MACROS_LIST, COLORES_ESTADO, NOMBRES_ESTADO,
    COLORES_REGIMEN, NOMBRES_REGIMEN, COLORES_ALERTA, PESOS_PIB,
    cargar_dashboard_data, cargar_panel, cargar_icds_star,
    cargar_icds_agg, cargar_markov, cargar_markov_params,
    cargar_leontief, simular_choque,
)

st.set_page_config(
    page_title="MVES-CO · Alerta Temprana",
    page_icon="🔔", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""<style>
[data-testid="stSidebar"]{background:#1F4E79}
[data-testid="stSidebar"] *{color:white!important}
.stMetric label{font-size:12px!important}
</style>""", unsafe_allow_html=True)


def hex_to_rgba(hex_color, alpha=0.6):
    """Convierte colores HEX (#RRGGBB) a rgba válido para Plotly."""
    try:
        h = str(hex_color).strip().lstrip("#")
        if len(h) != 6:
            return "rgba(136,136,136,{})".format(alpha)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    except Exception:
        return "rgba(136,136,136,{})".format(alpha)

def mkp_get(key, default="N/D"):
    """Obtiene parámetros de Markov sin romper la app si falta una llave."""
    try:
        return mkp.get(key, default) if isinstance(mkp, dict) else default
    except Exception:
        return default

# Cargar datos
@st.cache_data
def get_data():
    D    = cargar_dashboard_data()
    panel = cargar_panel()
    star  = cargar_icds_star()
    agg   = cargar_icds_agg()
    mk    = cargar_markov()
    mkp   = cargar_markov_params()
    L, A  = cargar_leontief()
    return D, panel, star, agg, mk, mkp, L, A

D, panel, df_star, agg, df_mk, mkp, L, A = get_data()

ult_mes       = D['ultimo_mes']
icds_agg_ult  = D['icds_agg_actual']
reg_ult       = D['regimen_icds']
ult_sectores  = D['ultimo']
mk_alerta     = mkp_get('alerta_actual', 'VERDE')
mk_prob       = float(mkp_get('prob_crisis_actual', 0) or 0)

with st.sidebar:
    st.markdown("## 🔔 MVES-CO")
    st.markdown("**Sistema de Alerta Temprana**\nContagio Financiero · Colombia")
    st.divider()
    pagina = st.radio("Navegación", [
        "📊 Resumen ejecutivo",
        "📈 Sectores ICDS / ICDS*",
        "⏱ Evolución temporal",
        "🔄 Markov — Riesgo financiero",
        "🔗 Red de contagio MIP",
        "💥 Simulador de choque",
        "📖 Metodología",
    ])
    st.divider()
    st.markdown(f"**Último mes:** `{ult_mes}`")
    st.markdown(f"**ICDS* sistémico:** `{icds_agg_ult:.4f}`")
    c = {"VERDE":"🟢","AMARILLA":"🟡","ROJA":"🔴"}
    st.markdown(f"**Alerta financiera:** {c[mk_alerta]} `{mk_alerta}`")
    st.caption("Katy Pacheco Manchego\nJulian Andres Firacative Varon\nUniversidad Central · 2026")

# ═══════════════════════════════ RESUMEN ══════════════════════════
if pagina == "📊 Resumen ejecutivo":
    st.title("📊 Resumen Ejecutivo")
    st.caption(f"Sistema de Alerta Temprana — Contagio Financiero Intersectorial · Colombia · {ult_mes}")

    c1,c2,c3,c4,c5 = st.columns(5)
    s45 = [m for m in MACROS_LIST if ult_sectores.get(m,{}).get('estado_star','') in ['S4','S5']]
    mip_alert = [m for m in MACROS_LIST if ult_sectores.get(m,{}).get('ajuste_mip',0) > 0.015]
    c1.metric("ICDS* sistémico", f"{icds_agg_ult:.4f}")
    c2.metric("Régimen ICDS", NOMBRES_REGIMEN[reg_ult])
    c3.metric("Alerta Markov", mk_alerta, delta=f"P(crisis)={mk_prob*100:.1f}%", delta_color="inverse" if mk_prob>0.4 else "off")
    c4.metric("Sectores S4-S5", f"{len(s45)} / 12")
    c5.metric("Con contagio MIP", f"{len(mip_alert)} / 12")

    st.divider()
    col1, col2 = st.columns([1.1,0.9])
    with col1:
        st.subheader("Estado actual — 12 macrosectores")
        sorted_m = sorted(MACROS_LIST, key=lambda m: ult_sectores.get(m,{}).get('icds_star',0), reverse=True)
        for m in sorted_m:
            d = ult_sectores.get(m,{}); est = d.get('estado_star','S4'); v = d.get('icds_star',0)
            color = COLORES_ESTADO.get(est,'#888')
            aj = d.get('ajuste_mip',0); alerta = "⚠️" if aj>0.015 else ""
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;'
                f'border-bottom:1px solid #e1e8f0;font-size:13px">'
                f'<span style="background:{color}22;color:{color};font-weight:600;'
                f'padding:1px 8px;border-radius:20px;font-size:11px">{est}</span>'
                f'<span style="flex:1">{MACROS[m]}</span>'
                f'<div style="width:90px;background:#f0f4f8;border-radius:3px;height:6px">'
                f'<div style="width:{int(v*100)}%;height:6px;background:{color};border-radius:3px"></div></div>'
                f'<span style="font-weight:600;min-width:44px;text-align:right">{v:.3f}</span>'
                f'<span style="font-size:11px;color:#E24B4A;min-width:18px">{alerta}</span></div>',
                unsafe_allow_html=True)

    with col2:
        st.subheader("Distribución de estados")
        dist = {}
        for m in MACROS_LIST:
            e = ult_sectores.get(m,{}).get('estado_star','S4')
            dist[e] = dist.get(e,0)+1
        fig = go.Figure(go.Pie(
            labels=[f"{k} {NOMBRES_ESTADO[k]}" for k in dist],
            values=list(dist.values()),
            marker_colors=[COLORES_ESTADO[k] for k in dist],
            hole=0.45, textinfo="value+percent", textfont_size=12,
        ))
        fig.update_layout(height=250, margin=dict(t=0,b=0,l=0,r=0), legend=dict(font_size=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Doble señal de alerta")
        col_a, col_b = st.columns(2)
        col_a.metric("ICDS* — Régimen", NOMBRES_REGIMEN[reg_ult],
                     help="Basado en el ICDS* sectorial agregado")
        col_b.metric("Markov financiero", mk_alerta,
                     help=f"P(Crisis)={mk_prob*100:.1f}% | Score_Riesgo TRM+TPM")
        st.caption("El modelo combina dos perspectivas: vulnerabilidad sectorial (ICDS*) y riesgo financiero sistémico (Markov sobre TRM+Tasa).")

# ═══════════════════════════════ SECTORES ═════════════════════════
elif pagina == "📈 Sectores ICDS / ICDS*":
    st.title("📈 Clasificación Sectorial — ICDS vs ICDS*")
    st.caption(f"Último mes: {ult_mes} · ICDS con 5 dimensiones: ISE + Empleo + Costos + TPM + TRM")

    col1,col2 = st.columns([1.2,0.8])
    with col1:
        tabla = []
        for m in sorted(MACROS_LIST, key=lambda x: ult_sectores.get(x,{}).get('icds_star',0), reverse=True):
            d = ult_sectores.get(m,{})
            tabla.append({"Macrosector":MACROS[m],
                          "ICDS":d.get('icds',0), "ICDS*":d.get('icds_star',0),
                          "Ajuste MIP":d.get('ajuste_mip',0),
                          "Estado":d.get('estado',''), "Estado*":d.get('estado_star','')})
        df_t = pd.DataFrame(tabla)
        def ce(v):
            return {"S1":"background-color:#EAF3DE","S2":"background-color:#E1F5EE",
                    "S3":"background-color:#E6F1FB","S4":"background-color:#FAEEDA",
                    "S5":"background-color:#FCEBEB"}.get(v,"")
        st.dataframe(
            df_t.style.applymap(ce, subset=["Estado","Estado*"])
            .background_gradient(subset=["ICDS","ICDS*"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({"ICDS":"{:.4f}","ICDS*":"{:.4f}","Ajuste MIP":"{:.4f}"}),
            use_container_width=True, height=450)

    with col2:
        sorted2 = sorted(MACROS_LIST, key=lambda m: ult_sectores.get(m,{}).get('icds_star',0))
        labels2 = [MACROS[m][:20] for m in sorted2]
        v_icds = [ult_sectores.get(m,{}).get('icds',0) for m in sorted2]
        v_star = [ult_sectores.get(m,{}).get('icds_star',0) for m in sorted2]
        colors2 = [COLORES_ESTADO.get(ult_sectores.get(m,{}).get('estado_star','S3'),'#888') for m in sorted2]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=labels2,x=v_icds,orientation="h",name="ICDS",
                               marker_color="rgba(55,138,221,.4)",marker_line_color="#378ADD",marker_line_width=1))
        fig2.add_trace(go.Bar(y=labels2,x=v_star,orientation="h",name="ICDS*",
                               marker_color=[hex_to_rgba(c, 0.60) for c in colors2],
                               marker_line_color=colors2,marker_line_width=1.5))
        for x in [0.40,0.60]:
            fig2.add_vline(x=x,line_dash="dot",line_color="gray",opacity=0.4)
        fig2.update_layout(barmode="overlay",height=430,
                            legend=dict(orientation="h",y=1.05,font_size=11),
                            xaxis=dict(range=[0,1],title="ICDS"),
                            margin=dict(t=10,b=10,l=5,r=5))
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Heatmap ICDS* histórico (2019–presente)")
    fechas_h = [f for f in sorted(df_star['fecha'].unique()) if f>='2019-01'][::2]
    pivot = df_star.pivot_table(index='macrosector_id',columns='fecha',values='icds_star')
    pivot = pivot[[c for c in pivot.columns if c in fechas_h]]
    pivot.index = [MACROS[m][:22] for m in pivot.index]
    fig_h = px.imshow(pivot, color_continuous_scale="RdYlGn", zmin=0, zmax=1,
                       labels=dict(x="Mes",y="Macrosector",color="ICDS*"), aspect="auto", text_auto=".2f")
    fig_h.update_layout(height=370, margin=dict(t=10,b=10,l=10,r=10))
    fig_h.update_xaxes(tickangle=35, tickfont_size=9)
    fig_h.update_traces(textfont_size=8)
    st.plotly_chart(fig_h, use_container_width=True)

# ═══════════════════════════════ EVOLUCIÓN ════════════════════════
elif pagina == "⏱ Evolución temporal":
    st.title("⏱ Evolución Temporal del ICDS")
    c1,c2,c3 = st.columns(3)
    m_sel = c1.selectbox("Sector", MACROS_LIST, format_func=lambda m: f"{m} — {MACROS[m]}")
    ind   = c2.selectbox("Indicador", ["icds","icds_star"],
                          format_func=lambda x: "ICDS original" if x=="icds" else "ICDS* (ajustado MIP)")
    desde = c3.selectbox("Desde", ["2015-01","2018-01","2019-01","2022-01"])

    sub = df_star[(df_star['macrosector_id']==m_sel)&(df_star['fecha']>=desde)].sort_values('fecha')
    fig_e = go.Figure()
    for ymin,ymax,color,label in [(0.8,1,"#639922","S1"),(0.6,0.8,"#1D9E75","S2"),
                                    (0.4,0.6,"#378ADD","S3"),(0.2,0.4,"#BA7517","S4"),(0,0.2,"#E24B4A","S5")]:
        fig_e.add_hrect(y0=ymin,y1=ymax,fillcolor=color,opacity=0.06,line_width=0,
                         annotation_text=label,annotation_position="right",annotation_font_size=10)
    for y in [0.2,0.4,0.6,0.8]:
        fig_e.add_hline(y=y,line_dash="dot",line_color="gray",opacity=0.4,line_width=0.8)
    fig_e.add_vrect(x0="2020-03",x1="2021-12",fillcolor="#E24B4A",opacity=0.07,line_width=0,
                    annotation_text="COVID-19",annotation_font_size=9)
    fig_e.add_trace(go.Scatter(x=sub['fecha'],y=sub[ind],mode='lines',
                                line=dict(color='#1F4E79',width=2),
                                fill='tozeroy',fillcolor='rgba(55,138,221,.07)',
                                name='ICDS*' if ind=='icds_star' else 'ICDS'))
    fig_e.update_layout(title=f"{MACROS[m_sel]} — {'ICDS*' if ind=='icds_star' else 'ICDS'}",
                         yaxis=dict(range=[0,1],title="Índice"),height=340,
                         margin=dict(t=40,b=20,l=10,r=80))
    st.plotly_chart(fig_e, use_container_width=True)

    st.subheader("Comparar sectores")
    secs = st.multiselect("Sectores",MACROS_LIST,default=["M03","M05","M08"],
                           format_func=lambda m: f"{m} — {MACROS[m]}")
    if secs:
        fig_c = go.Figure()
        pal = px.colors.qualitative.Set2
        for i,m in enumerate(secs):
            s = df_star[(df_star['macrosector_id']==m)&(df_star['fecha']>=desde)].sort_values('fecha')
            fig_c.add_trace(go.Scatter(x=s['fecha'],y=s['icds_star'],mode='lines',
                                        name=MACROS[m][:25],line=dict(color=pal[i%len(pal)],width=1.8)))
        for y in [0.2,0.4,0.6,0.8]:
            fig_c.add_hline(y=y,line_dash="dot",line_color="gray",opacity=0.3,line_width=0.7)
        fig_c.add_vrect(x0="2020-03",x1="2021-12",fillcolor="#E24B4A",opacity=0.07,line_width=0)
        fig_c.update_layout(yaxis=dict(range=[0,1],title="ICDS*"),height=310,
                             legend=dict(orientation="h",y=-0.25,font_size=10),
                             margin=dict(t=20,b=10,l=10,r=10))
        st.plotly_chart(fig_c, use_container_width=True)

# ═══════════════════════════════ MARKOV ═══════════════════════════
elif pagina == "🔄 Markov — Riesgo financiero":
    st.title("🔄 Modelo Markov-Switching — Riesgo Financiero Sistémico")
    st.caption("MarkovAutoregression AR(1) sobre ICDS* sistémico | switching_ar=True | 2 regímenes | statsmodels | 50 semillas")

    # Métricas Markov
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("P(Crisis) actual", f"{float(mkp_get('prob_crisis_actual', 0) or 0)*100:.1f}%")
    c2.metric("Alerta actual", mkp_get('alerta_actual', 'VERDE'))
    c3.metric("Alertas ROJAS (históricas)", mkp_get('n_rojas', 0))
    c4.metric("Validación COVID", f"{float(mkp_get('covid_p_crisis', 0) or 0)*100:.1f}%",
              help="P(Crisis) promedio durante mar-sep 2020")

    st.divider()

    # Gráfica principal: Prob_Crisis con alertas
    fig_mk = make_subplots(rows=2,cols=1,shared_xaxes=True,
                            row_heights=[0.65,0.35],vertical_spacing=0.04)

    # Zonas de alerta
    fig_mk.add_hrect(y0=0.70,y1=1.05,fillcolor="#E24B4A",opacity=0.08,line_width=0,row=1,col=1)
    fig_mk.add_hrect(y0=0.40,y1=0.70,fillcolor="#BA7517",opacity=0.08,line_width=0,row=1,col=1)
    fig_mk.add_hrect(y0=0,y1=0.40,fillcolor="#639922",opacity=0.06,line_width=0,row=1,col=1)

    # Serie Prob_Crisis
    fechas_mk = pd.to_datetime(df_mk['Periodo']).dt.to_period('M').astype(str)
    fig_mk.add_trace(go.Scatter(x=df_mk['Periodo'],y=df_mk['Prob_Crisis'],
                                 mode='lines',line=dict(color='#1F4E79',width=1.8),
                                 name='P(Crisis)',fill='tozeroy',fillcolor='rgba(31,78,121,.07)'),row=1,col=1)
    # Puntos de alerta
    for nivel,color,umbral_min,umbral_max in [
        ('ROJA','#E24B4A',0.70,1.01),('AMARILLA','#BA7517',0.40,0.70),('VERDE','#639922',0,0.40)]:
        mask = (df_mk['Alerta']==nivel) if 'Alerta' in df_mk.columns else (df_mk['Prob_Crisis']>=umbral_min)&(df_mk['Prob_Crisis']<umbral_max)
        sub_mk = df_mk[mask]
        fig_mk.add_trace(go.Scatter(x=sub_mk['Periodo'],y=sub_mk['Prob_Crisis'],
                                     mode='markers',name=f'Alerta {nivel}',
                                     marker=dict(color=color,size=5,opacity=0.8)),row=1,col=1)
    for y,color,label in [(0.70,'#E24B4A','Umbral Roja 70%'),(0.40,'#BA7517','Umbral Amarilla 40%')]:
        fig_mk.add_hline(y=y,line_dash="dash",line_color=color,opacity=0.6,
                          annotation_text=label,annotation_font_size=9,row=1,col=1)

    # Score_Riesgo en panel inferior
    fig_mk.add_trace(go.Scatter(x=df_mk['Periodo'],y=df_mk['Score_Riesgo'],
                                 mode='lines',line=dict(color='#2E75B6',width=1.2),
                                 name='Score Riesgo'),row=2,col=1)
    fig_mk.update_layout(height=520,margin=dict(t=20,b=20,l=10,r=10),
                          title="Probabilidad de Crisis — Modelo Markov-Switching (Score_Riesgo: TRM+TPM+Volatilidades)",
                          yaxis=dict(range=[0,1.05],title="P(Crisis)"),
                          yaxis2=dict(title="Score Riesgo"),
                          legend=dict(orientation="h",y=1.05,font_size=10))
    st.plotly_chart(fig_mk, use_container_width=True)

    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Parámetros del modelo")
        st.markdown(f"""
        | Parámetro | Valor |
        |-----------|-------|
        | Regímenes (k) | 2 (Calma / Crisis) |
        | Observaciones | {mkp_get('n_obs')} meses |
        | Período | {mkp_get('periodo_min', df_mk['Periodo'].min() if 'Periodo' in df_mk.columns else 'N/D')} — {mkp_get('periodo_max', df_mk['Periodo'].max() if 'Periodo' in df_mk.columns else 'N/D')} |
        | AIC | {mkp_get('aic')} |
        | Log-likelihood | {mkp_get('llf')} |
        | Variable | ICDS* sistémico agregado (ponderado PIB) |
        | Implementación | statsmodels.MarkovAutoregression |
        """)
        st.subheader("Validación")
        covid_p_crisis = float(mkp_get('covid_p_crisis', 0) or 0)
        cov_ok = covid_p_crisis > 0.60
        if cov_ok:
            st.success(f"✓ COVID-19 detectado: P(crisis) = {covid_p_crisis*100:.1f}% (mar-sep 2020)")
        else:
            st.warning("⚠ Revisar detección COVID")

    with col2:
        st.subheader("TRM y tasa de política")
        fig_trm = go.Figure()
        fig_trm.add_trace(go.Scatter(x=df_mk['Periodo'],y=df_mk['TRM'],
                                      mode='lines',name='TRM (COP/USD)',
                                      line=dict(color='#BA7517',width=1.5),yaxis='y'))
        fig_trm.add_trace(go.Scatter(x=df_mk['Periodo'],y=df_mk['Tasa'],
                                      mode='lines',name='TPM (%)',
                                      line=dict(color='#1F4E79',width=1.5),yaxis='y2'))
        fig_trm.update_layout(height=280,margin=dict(t=20,b=10,l=10,r=60),
                               legend=dict(font_size=10,orientation="h",y=1.1),
                               yaxis=dict(title="TRM",titlefont_color="#BA7517"),
                               yaxis2=dict(title="TPM (%)",overlaying="y",side="right",
                                           titlefont_color="#1F4E79"))
        st.plotly_chart(fig_trm, use_container_width=True)

# ═══════════════════════════════ MIP ══════════════════════════════
elif pagina == "🔗 Red de contagio MIP":
    st.title("🔗 Red de Contagio — Matriz Insumo-Producto")
    st.caption("Inversa de Leontief L = (I-A)⁻¹ | DANE MIP 2021 | 12 macrosectores")

    col1,col2 = st.columns([1.1,0.9])
    with col1:
        umbral = st.slider("Umbral vínculo A_ij", 0.01,0.15,0.04,0.01)
        A_np = A.values.copy(); np.fill_diagonal(A_np,0)
        G = nx.DiGraph()
        for n in MACROS_LIST: G.add_node(n)
        for i,o in enumerate(MACROS_LIST):
            for j,d2 in enumerate(MACROS_LIST):
                if A_np[i,j]>umbral: G.add_edge(o,d2,weight=A_np[i,j])
        pos = nx.circular_layout(G,scale=2.5)
        colores_n = px.colors.qualitative.Set3[:12]
        mult_d = D['leontief_diag']
        ex=[]; ey=[]
        for u,v in G.edges():
            x0,y0=pos[u]; x1,y1=pos[v]
            ex+=[x0,x1,None]; ey+=[y0,y1,None]
        fig_n = go.Figure()
        fig_n.add_trace(go.Scatter(x=ex,y=ey,mode='lines',line=dict(color='rgba(100,100,100,.3)',width=1.5),hoverinfo='none'))
        fig_n.add_trace(go.Scatter(
            x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
            mode='markers+text',
            marker=dict(size=[18+35*mult_d.get(n,1) for n in G.nodes()],color=colores_n,line=dict(width=1.5,color='white')),
            text=list(G.nodes()), textposition='middle center', textfont=dict(size=9,family='Arial Black'),
            hovertext=[f"{n}: {MACROS[n]}<br>Mult={mult_d.get(n,0):.3f}" for n in G.nodes()],
            hoverinfo='text'))
        fig_n.update_layout(showlegend=False,height=420,margin=dict(t=20,b=20,l=20,r=20),
                             xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                             yaxis=dict(showgrid=False,zeroline=False,showticklabels=False))
        st.plotly_chart(fig_n, use_container_width=True)

    with col2:
        st.subheader("Multiplicadores de Leontief")
        md_df = pd.DataFrame({'Macrosector':[MACROS[m] for m in MACROS_LIST],
                               'Multiplicador':[round(mult_d.get(m,1),4) for m in MACROS_LIST]})
        md_df = md_df.sort_values('Multiplicador',ascending=False)
        fig_m = go.Figure(go.Bar(y=md_df['Macrosector'].str[:20],x=md_df['Multiplicador'],
                                  orientation='h',marker_color='#2E75B6',marker_opacity=0.8))
        fig_m.add_vline(x=1,line_dash='dash',line_color='gray',opacity=0.5)
        fig_m.update_layout(height=340,margin=dict(t=10,b=10,l=5,r=10),xaxis_title="L_ii")
        st.plotly_chart(fig_m, use_container_width=True)

        st.subheader("Top vínculos")
        A_off = A.copy(); np.fill_diagonal(A_off.values,0)
        pares = sorted([(float(A_off.loc[i,j]),i,j) for i in MACROS_LIST for j in MACROS_LIST
                         if float(A_off.loc[i,j])>0.02], reverse=True)
        df_p = pd.DataFrame([(MACROS[i][:18],MACROS[j][:18],round(v,4)) for v,i,j in pares[:10]],
                             columns=["Proveedor","Comprador","A_ij"])
        st.dataframe(df_p, use_container_width=True, hide_index=True)

# ═══════════════════════════════ CHOQUE ═══════════════════════════
elif pagina == "💥 Simulador de choque":
    st.title("💥 Simulador de Choque Sectorial")
    c1,c2 = st.columns([1,2])
    with c1:
        sec_c = st.selectbox("Sector en crisis",MACROS_LIST,index=4,format_func=lambda m:f"{m} — {MACROS[m]}")
        icds_c = st.slider("ICDS del sector",0.00,0.39,0.10,0.01)
        st.metric("Estado simulado","S5 Contracción" if icds_c<0.20 else "S4 Decrecimiento")
        st.markdown("---")
        st.markdown("""**Fórmula de contagio:**
```
ICDS*_i = ICDS_i
  − 0.5 × Σ_j [A_ji × max(0, 0.5 − ICDS_j)]
```
""")
    with c2:
        res = simular_choque(df_star, A, sec_c, icds_c)
        fig_ch = go.Figure()
        fig_ch.add_trace(go.Bar(y=res['macrosector'].str[:20],x=res['icds_base'],orientation='h',
                                 name='ICDS base',marker_color='rgba(55,138,221,.5)',
                                 marker_line_color='#378ADD',marker_line_width=1))
        fig_ch.add_trace(go.Bar(y=res['macrosector'].str[:20],x=res['icds_shock'],orientation='h',
                                 name='ICDS shock',marker_color='rgba(226,75,74,.5)',
                                 marker_line_color='#E24B4A',marker_line_width=1))
        fig_ch.add_vline(x=0.4,line_dash='dash',line_color='#BA7517',opacity=0.5)
        fig_ch.update_layout(barmode='overlay',height=400,
                              title=f"Choque: {MACROS[sec_c]} → ICDS={icds_c:.2f}",
                              xaxis=dict(range=[0,1],title='ICDS'),
                              legend=dict(orientation='h',y=1.1,font_size=11),
                              margin=dict(t=45,b=10,l=5,r=10))
        st.plotly_chart(fig_ch, use_container_width=True)
    res['Alerta'] = res['caida'].apply(lambda v: "⚠️ CONTAGIO" if v>0.02 else ("🔴 EPICENTRO" if v==res['caida'].max() else "—"))
    st.dataframe(res[['macrosector','icds_base','icds_shock','caida','Alerta']]
                 .rename(columns={'macrosector':'Sector','icds_base':'ICDS base','icds_shock':'ICDS shock','caida':'Caída'})
                 .style.background_gradient(subset=['Caída'],cmap='Reds')
                 .format({'ICDS base':'{:.4f}','ICDS shock':'{:.4f}','Caída':'{:.4f}'}),
                 use_container_width=True, hide_index=True)

# ═══════════════════════════════ METODOLOGÍA ══════════════════════
elif pagina == "📖 Metodología":
    st.title("📖 Metodología — Sistema MVES-CO")
    st.markdown(f"""
## Arquitectura del modelo integrado

El sistema combina **tres capas analíticas** que responden al objetivo de su sustentación:
*"¿Cómo puede un sistema analítico que combine técnicas de estadística descriptiva y modelado predictivo mejorar la gestión del riesgo de contagio financiero?"*

---

### Capa 1 — ICDS: Índice de Condiciones por Dimensión Sectorial

$$ICDS_i = \\sum_d \\frac{{w_d \\cdot norm_{{d,i}}}}{{\\sum_d w_d}}$$

**Pesos obtenidos por Análisis de Componentes Principales (ACP) — cargas absolutas PC1:**

| Dimensión | Variable | Peso ACP | Fuente |
|-----------|----------|----------|--------|
| Tasa de política monetaria (inv) | TPM | **35,26%** | BanRep — real hasta abr 2026 |
| Variación de costos sectoriales (inv) | IPP / IPC | **32,83%** | DANE — hasta abr 2026 |
| Actividad económica | ISE | **13,57%** | DANE — hasta mar 2026 |
| Variación de empleo | GEIH | **12,95%** | DANE — hasta mar 2026 |
| Tipo de cambio ajustado sectorial | TRM | **5,39%** | BanRep — hasta abr 2026 |

**Normalización:** Probability Integral Transform (PIT) histórica, excluyendo 2020–2021 (COVID crash).

---

### Capa 2 — ICDS*: Ajuste por contagio MIP (Leontief)

$$ICDS^*_i = ICDS_i - 0.5 \\sum_j A_{{ji}} \\cdot \\max(0,\\ 0.5 - ICDS_j)$$

**Fuente:** MIP DANE 2021. M02 (Minería) y M04 (Electricidad) comparten proxy GEIH.

---

### Capa 3 — Markov-Switching AR(1): Detección de régimen sistémico

$$ICDS^*_t = \\mu_{{s_t}} + \\phi_{{s_t}} \\cdot ICDS^*_{{t-1}} + \\sigma \\cdot \\varepsilon_t, \\quad s_t \\sim Markov(P)$$

$$P(s_t = j \\mid s_{{t-1}} = i) = p_{{ij}}$$

- **Variable de entrada:** ICDS* sistémico agregado (ponderado por PIB)
- **2 regímenes no observados:** R1 Expansión (≥ 0,60) / R2 Estabilidad (0,40–0,60)
- **Especificación:** switching_ar = True (φ cambia por régimen), switching_variance = False
- **Estimación:** statsmodels.MarkovAutoregression — búsqueda con 50 semillas aleatorias
- **Validación:** COVID-19 detectado con P(crisis) = {float(mkp_get('covid_p_crisis', 0) or 0)*100:.1f}% (mar-sep 2020)
- **AIC:** {mkp_get('aic')} | **Log-likelihood:** {mkp_get('llf')}

---

### Referencias clave
- Hamilton, J.D. (1989). *A new approach to the economic analysis of nonstationary time series.*
- Leontief, W. (1941). *The Structure of American Economy.*
- Nardo, M. et al. (2005). *Tools for Composite Indicators Building.* JRC-EC.
- OCDE (2008). *Handbook on Constructing Composite Indicators.*
- Reinhart, C. & Rogoff, K. (2009). *This Time Is Different.* Princeton UP.
""")
    col1,col2 = st.columns(2)
    with col1:
        st.subheader("Fuentes de datos")
        st.markdown("""| Fuente | Variable | Hasta |
|--------|----------|-------|
| DANE — ISE 12 actividades | ΔActividad | Feb 2026 |
| DANE — GEIH Ocupados | ΔEmpleo | Abr 2026 |
| DANE — IPP Producción | ΔCostos | Mar 2026 |
| BanRep — IPC | Inflación | Mar 2026 |
| BanRep — TPM | Tasa política | Abr 2026 |
| BanRep — TRM | Tipo de cambio | Abr 2026 |
| DANE — MIP 2021 | Coef. Leontief | Anual |""")
        st.caption("TPM real BanRep hasta abril 2026 (Tasa_de_politicamonteraria.xlsx). MIP DANE 2021 (última disponible).")
    with col2:
        st.subheader("Limitaciones reconocidas")
        st.markdown("""
- MIP estática (2021): no captura cambios estructurales post-COVID; próxima actualización DANE pendiente
- M02 (Minería) y M04 (Electricidad) comparten fila GEIH "Suministro electricidad, gas, agua" como proxy
- ISE disponible con rezago de un mes; se aplica forward-fill de un período para el mes de cierre
- El Markov AR(1) opera sobre ICDS* agregado; no captura heterogeneidad intra-régimen
- Agregación a 12 macrosectores reduce heterogeneidad interna de sectores compuestos
""")
