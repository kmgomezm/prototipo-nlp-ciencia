"""Prototipo de visualización: la ciencia en columnas de opinión colombianas (2018–2020)."""
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

TAU = 0.47
DATA = Path(__file__).parent / "data"
TIPOS = {"ORG": "Organizaciones", "PER": "Personas", "LOC": "Lugares"}

st.set_page_config(page_title="Ciencia en columnas de opinión", layout="wide")


# ---------------- Datos ----------------
@st.cache_data
def cargar():
    docs = pd.read_parquet(DATA / "documentos.parquet")
    frag = pd.read_parquet(DATA / "fragmentos.parquet")
    ent = pd.read_parquet(DATA / "entidades.parquet")
    top = pd.read_parquet(DATA / "topicos_info.parquet")

    meta = docs[["id_doc", "periodico", "autor", "fecha", "mes", "titulo", "url"]]
    frag = frag.merge(meta, on="id_doc", how="left")
    frag["es_ciencia"] = frag["sim_max"] >= TAU
    frag["topico_etq"] = frag["topico"].map(dict(zip(top["topico"], top["etiqueta"])))

    ent = ent.merge(frag[["uid", "id_doc", "es_ciencia", "categoria"]], on="uid", how="left")
    return docs, frag, ent, top


def tabla_fragmentos(df, key, max_filas=300):
    """Tabla común de fragmentos con enlace a la columna original."""
    total = len(df)
    st.caption(f"{total:,} fragmentos" + (f" · se muestran los {max_filas} de mayor similitud" if total > max_filas else ""))
    vista = (df.sort_values("sim_max", ascending=False).head(max_filas)
            [["fecha", "periodico", "autor", "titulo", "categoria", "topico_etq", "sim_max", "texto_corto"]])
    st.dataframe(
        vista, key=key, hide_index=True, use_container_width=True,
        column_config={
            "fecha": st.column_config.DateColumn("Fecha", format="YYYY-MM-DD"),
            "periodico": "Periódico",
            "autor": "Autor",
            "titulo": "Título",
            "categoria": "Categoría UNESCO",
            "topico_etq": "Tópico",
            "sim_max": st.column_config.NumberColumn("Similitud", format="%.3f"),
            "texto_corto": st.column_config.TextColumn("Fragmento (recortado)", width="large"),
        },
    )


docs, frag, ent, top = cargar()
etiquetas = dict(zip(top["topico"], top["etiqueta"]))

# ---------------- Barra lateral ----------------
st.sidebar.header("Filtros")
meses = sorted(pd.to_datetime(docs["mes"].unique()))
ini, fin = st.sidebar.select_slider(
    "Periodo", options=meses, value=(meses[0], meses[-1]),
    format_func=lambda m: m.strftime("%Y-%m"),
)
periodicos = sorted(docs["periodico"].unique())
sel_per = st.sidebar.multiselect("Periódico", periodicos, default=periodicos)
categorias = sorted(frag.loc[frag["es_ciencia"], "categoria"].unique())
sel_cat = st.sidebar.multiselect("Categoría UNESCO", categorias, default=categorias)
st.sidebar.caption(f"Umbral de clasificación fijo: τ = {TAU}")

docs_f = docs[docs["mes"].between(ini, fin) & docs["periodico"].isin(sel_per)]
if docs_f.empty or not sel_cat:
    st.warning("Los filtros no dejan datos. Ajusta el periodo, los periódicos o las categorías.")
    st.stop()

frag_f = frag[frag["id_doc"].isin(docs_f["id_doc"])]
frag_c = frag_f[frag_f["es_ciencia"] & frag_f["categoria"].isin(sel_cat)]
docs_con = set(frag_c["id_doc"])

# ---------------- Encabezado ----------------
st.title("La ciencia en columnas de opinión colombianas (2018–2020)")
st.caption(
    "Prototipo de exploración de menciones a la ciencia en 13.676 columnas de El Espectador, "
    "El Tiempo y Semana, identificadas mediante búsqueda semántica con las categorías del Tesauro de la UNESCO."
)
t1, t2, t3, t4, t5 = st.tabs(["Panorama", "Tópicos", "Entidades", "Autores", "Fragmentos"])

# ---------------- Panorama ----------------
with t1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Columnas", f"{len(docs_f):,}")
    c2.metric("Columnas con mención de ciencia", f"{len(docs_con):,}",
              f"{len(docs_con) / len(docs_f):.1%} del total", delta_color="off")
    c3.metric("Fragmentos de ciencia", f"{len(frag_c):,}")

    serie = (docs_f.assign(con=docs_f["id_doc"].isin(docs_con))
                   .groupby("mes")["con"].mean().mul(100).reset_index(name="pct"))
    fig = px.line(serie, x="mes", y="pct", markers=True,
                  labels={"mes": "Mes", "pct": "% de columnas con mención de ciencia"},
                  title="Porcentaje mensual de columnas con menciones de ciencia")
    pico = serie.loc[serie["pct"].idxmax()]
    fig.add_annotation(x=pico["mes"], y=pico["pct"],
                       text=f"Pico: {pico['mes']:%Y-%m} ({pico['pct']:.1f} %)", showarrow=True)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    cat = frag_c["categoria"].value_counts().reset_index()
    cat.columns = ["categoria", "n"]
    fig_cat = px.bar(cat, x="n", y="categoria", orientation="h",
                     labels={"n": "Fragmentos", "categoria": ""},
                     title="Fragmentos de ciencia por categoría UNESCO")
    fig_cat.update_layout(yaxis={"categoryorder": "total ascending"}, height=550)
    col_a.plotly_chart(fig_cat, use_container_width=True)

    per = (docs_f.assign(con=docs_f["id_doc"].isin(docs_con))
                 .groupby("periodico")["con"].mean().mul(100).reset_index(name="pct"))
    fig_per = px.bar(per, x="periodico", y="pct",
                     labels={"periodico": "", "pct": "% de columnas con mención de ciencia"},
                     title="Columnas con menciones de ciencia por periódico")
    col_b.plotly_chart(fig_per, use_container_width=True)

# ---------------- Tópicos ----------------
with t2:
    st.caption(
        "Tópicos obtenidos con una reejecución de BERTopic con semilla fija sobre el subcorpus de ciencia "
        "(τ = 0,47). Conserva los ejes temáticos del artículo, aunque su agrupación difiere de la Tabla 11. "
        "Haz clic en una barra para ver el detalle del tópico."
    )
    cnt = frag_c.groupby("topico").size().rename("n").reset_index()
    dt = top.drop(columns=["n", "porcentaje"]).merge(cnt, on="topico", how="inner")
    dt["pct"] = dt["n"] / dt["n"].sum() * 100

    fig_t = px.bar(dt.sort_values("n"), x="pct", y="etiqueta", orientation="h",
                   custom_data=["topico"],
                   hover_data={"palabras": True, "coherencia": ":.3f", "n": True, "etiqueta": False},
                   labels={"pct": "% de fragmentos de ciencia", "etiqueta": ""},
                   title="Distribución de tópicos")
    fig_t.update_layout(height=550)
    ev = st.plotly_chart(fig_t, use_container_width=True, on_select="rerun",
                         selection_mode="points", key="bar_topicos")

    opciones = dt.sort_values("n", ascending=False)["topico"].tolist()
    elegido = None
    if ev and ev.selection and ev.selection.points:
        elegido = ev.selection.points[0]["customdata"][0]
    topico = st.selectbox("Tópico seleccionado", opciones,
                          index=opciones.index(elegido) if elegido in opciones else 0,
                          format_func=lambda t: etiquetas.get(t, str(t)))

    fila = dt[dt["topico"] == topico].iloc[0]
    m1, m2, m3 = st.columns(3)
    m1.metric("Fragmentos", f"{fila['n']:,}")
    m2.metric("% del subcorpus filtrado", f"{fila['pct']:.1f} %")
    m3.metric("Coherencia (c_v)", f"{fila['coherencia']:.3f}")
    st.markdown(f"**Palabras clave:** {fila['palabras']}")

    mensual = frag_c.groupby(["mes", "topico_etq"]).size().reset_index(name="n")
    fig_e = px.line(mensual, x="mes", y="n", color="topico_etq", markers=True,
                    labels={"mes": "Mes", "n": "Fragmentos", "topico_etq": "Tópico"},
                    title="Evolución mensual de los tópicos (clic en la leyenda para mostrar u ocultar)")
    visibles = set(dt.sort_values("n", ascending=False)["etiqueta"].head(5)) | {etiquetas[topico]}
    fig_e.for_each_trace(lambda tr: tr.update(visible=True if tr.name in visibles else "legendonly"))
    st.plotly_chart(fig_e, use_container_width=True)

    st.subheader("Fragmentos del tópico")
    tabla_fragmentos(frag_c[frag_c["topico"] == topico], key="frag_topico")

# ---------------- Entidades ----------------
with t3:
    c1, c2, c3 = st.columns(3)
    tipo = c1.radio("Tipo de entidad", list(TIPOS), format_func=TIPOS.get, horizontal=True)
    min_total = c2.number_input("Frecuencia total mínima", min_value=1, max_value=500, value=10)
    min_razon = c3.slider("Razón ciencia/total mínima", 0.0, 1.0, 0.30, 0.05)

    e = ent[(ent["tipo"] == tipo) & ent["id_doc"].isin(docs_f["id_doc"])]
    e_cien = e["es_ciencia"] & e["categoria"].isin(sel_cat)
    tabla = (e.assign(c=e_cien).groupby("entidad")
               .agg(total=("uid", "size"), ciencia=("c", "sum")).reset_index())
    tabla["razon"] = tabla["ciencia"] / tabla["total"]
    tabla = (tabla[(tabla["total"] >= min_total) & (tabla["razon"] >= min_razon)]
             .sort_values("ciencia", ascending=False).reset_index(drop=True))

    st.caption(
        "Menciones en todo el corpus filtrado frente a menciones en fragmentos de ciencia, ordenadas por "
        "frecuencia en ciencia. Con los valores por defecto se aplica el criterio del artículo (razón ≥ 30 %). "
        "Selecciona una fila para ver los fragmentos donde aparece."
    )
    sel = st.dataframe(
        tabla, key="tabla_ent", hide_index=True, use_container_width=True,
        on_select="rerun", selection_mode="single-row",
        column_config={
            "entidad": "Entidad",
            "total": "Menciones totales",
            "ciencia": "Menciones en ciencia",
            "razon": st.column_config.ProgressColumn("Razón ciencia/total", format="%.2f",
                                                     min_value=0.0, max_value=1.0),
        },
    )
    if sel.selection.rows:
        nombre = tabla.iloc[sel.selection.rows[0]]["entidad"]
        uids = e.loc[(e["entidad"] == nombre) & e_cien, "uid"].unique()
        st.subheader(f"Fragmentos de ciencia que mencionan «{nombre}»")
        tabla_fragmentos(frag_c[frag_c["uid"].isin(uids)], key="frag_entidad")

# ---------------- Autores ----------------
def moda(s):
    m = s.dropna().mode()
    return m.iat[0] if not m.empty else None

with t4:
    c1, c2 = st.columns(2)
    min_cols = c1.number_input("Columnas mínimas por autor", min_value=1, max_value=500, value=10)
    top_n = c2.slider("Autores en el gráfico de barras", 5, 30, 15)

    base = docs_f.assign(con=docs_f["id_doc"].isin(docs_con))
    aut = (base.groupby("autor")
               .agg(periodico=("periodico", lambda s: ", ".join(sorted(s.unique()))),
                    columnas=("id_doc", "size"),
                    con_ciencia=("con", "sum"))
               .reset_index())
    aut["pct"] = aut["con_ciencia"] / aut["columnas"] * 100
    fc = (frag_c.groupby("autor")
                .agg(fragmentos=("uid", "size"),
                     categoria=("categoria", moda),
                     topico=("topico_etq", moda))
                .reset_index())
    aut = aut.merge(fc, on="autor", how="left")
    aut["fragmentos"] = aut["fragmentos"].fillna(0).astype(int)
    aut = aut[aut["columnas"] >= min_cols].sort_values("pct", ascending=False).reset_index(drop=True)

    if aut.empty:
        st.info("Ningún autor cumple el mínimo de columnas con los filtros actuales.")
    else:
        pct_global = len(docs_con) / len(docs_f) * 100
        m1, m2, m3 = st.columns(3)
        m1.metric("Autores con el mínimo de columnas", f"{len(aut):,}")
        m2.metric("De ellos, con alguna mención de ciencia", f"{(aut['con_ciencia'] > 0).sum():,}")
        m3.metric("Mediana de % de columnas con ciencia", f"{aut['pct'].median():.1f} %")

        g1, g2 = st.columns(2)
        fig_s = px.scatter(
            aut, x="columnas", y="pct", color="periodico", hover_name="autor",
            custom_data=["autor"], log_x=True,
            hover_data={"columnas": True, "con_ciencia": True, "pct": ":.1f", "periodico": False},
            labels={"columnas": "Columnas publicadas (escala logarítmica)",
                    "pct": "% de columnas con mención de ciencia", "periodico": "Periódico",
                    "con_ciencia": "Columnas con ciencia"},
            title="Volumen de publicación frente a presencia de ciencia",
        )
        fig_s.add_hline(y=pct_global, line_dash="dash",
                        annotation_text=f"Promedio del corpus filtrado ({pct_global:.1f} %)")
        ev_s = g1.plotly_chart(fig_s, use_container_width=True, on_select="rerun",
                               selection_mode="points", key="disp_autores")

        top_aut = aut.head(top_n).sort_values("pct")
        fig_b = px.bar(top_aut, x="pct", y="autor", orientation="h",
                       hover_data={"columnas": True, "con_ciencia": True},
                       labels={"pct": "% de columnas con mención de ciencia", "autor": "",
                               "columnas": "Columnas", "con_ciencia": "Columnas con ciencia"},
                       title=f"Los {top_n} autores con mayor porcentaje")
        fig_b.update_layout(height=max(400, 28 * len(top_aut)))
        g2.plotly_chart(fig_b, use_container_width=True)

        st.caption(
            "Una columna cuenta como «con ciencia» si tiene al menos un fragmento clasificado como mención de "
            "ciencia con los filtros actuales. Selecciona una fila o haz clic en un punto del gráfico para ver "
            "los fragmentos del autor."
        )
        sel_a = st.dataframe(
            aut, key="tabla_autores", hide_index=True, use_container_width=True,
            on_select="rerun", selection_mode="single-row",
            column_config={
                "autor": "Autor",
                "periodico": "Periódico",
                "columnas": "Columnas",
                "con_ciencia": "Columnas con ciencia",
                "pct": st.column_config.ProgressColumn("% con ciencia", format="%.1f %%",
                                                       min_value=0.0, max_value=100.0),
                "fragmentos": "Fragmentos de ciencia",
                "categoria": "Categoría predominante",
                "topico": "Tópico predominante",
            },
        )

        autor_sel = None
        if ev_s and ev_s.selection and ev_s.selection.points:
            autor_sel = ev_s.selection.points[0]["customdata"][0]
        if sel_a.selection.rows:
            autor_sel = aut.iloc[sel_a.selection.rows[0]]["autor"]
        if autor_sel:
            st.subheader(f"Fragmentos de ciencia de {autor_sel}")
            tabla_fragmentos(frag_c[frag_c["autor"] == autor_sel], key="frag_autor")

# ---------------- Fragmentos ----------------
with t5:
    q = st.text_input("Buscar en los fragmentos (palabra o frase)")
    df = frag_c
    if q:
        df = df[df["texto_corto"].str.contains(q, case=False, na=False, regex=False)]
    tabla_fragmentos(df, key="frag_todos")
    st.caption(
        "Por derechos de autor, los textos se muestran recortados a 50 palabras. "
        "La búsqueda opera sobre ese recorte."
    )