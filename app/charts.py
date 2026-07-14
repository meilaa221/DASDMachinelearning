"""Plotly figure builders for the FastAPI dashboard.

Each function returns a ready-to-embed HTML <div> string (Plotly.js is loaded
once globally in base.html, so include_plotlyjs=False here to avoid loading
the ~1MB library repeatedly).
"""
import sys
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import NUMERIC_FEATURES, ORDINAL_ORDER, TARGET  # noqa: E402

# ---------------------------------------------------------------- Visual theme
# One shared Plotly template so every chart looks like part of the same
# product instead of the library's grey-gridline default.
GREEN_DARK = "#064E3B"
GREEN_MID = "#0E9F6E"
GREEN_SOFT = "#68DBA9"
GREEN_PALE = "#85F8C4"
COLORWAY = [GREEN_MID, "#3B82F6", "#F59E0B", GREEN_DARK, "#EC4899", GREEN_SOFT, "#8B5CF6", "#EF4444"]
KONDISI_COLORS = {"Buruk": "#EF4444", "Cukup": "#F59E0B", "Baik": GREEN_MID}

pio.templates["rumahku"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Inter, Segoe UI, sans-serif", size=12.5, color="#374151"),
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(font=dict(size=15, color="#111827", family="Inter", weight=700), x=0.02, xanchor="left",
                    y=0.97, yanchor="top", pad=dict(b=10)),
        margin=dict(l=55, r=25, t=65, b=70),
        xaxis=dict(showgrid=False, zeroline=False, showline=True, linecolor="#E5E7EB",
                    ticks="outside", tickcolor="#E5E7EB", tickfont=dict(color="#6B7280")),
        yaxis=dict(showgrid=True, gridcolor="#F3F4F6", zeroline=False, showline=False,
                    tickfont=dict(color="#6B7280")),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5,
                     font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#111827", font_size=12, font_family="Inter", font_color="white",
                          bordercolor="#111827"),
        colorscale=dict(
            sequential=[[0, "#F0FDF4"], [0.5, GREEN_SOFT], [1, GREEN_DARK]],
            diverging=[[0, "#DC2626"], [0.5, "#FFFFFF"], [1, GREEN_DARK]],
        ),
    )
)
pio.templates.default = "rumahku"
px.defaults.template = "rumahku"
px.defaults.color_discrete_sequence = COLORWAY

_counter = {"n": 0}
_CHART_CONFIG = {"displayModeBar": False, "responsive": True}


def _to_html(fig, height=360, hide_legend=False):
    _counter["n"] += 1
    if hide_legend:
        fig.update_layout(showlegend=False)
    fig.update_layout(autosize=True, height=height)
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config=_CHART_CONFIG,
        div_id=f"chart-{_counter['n']}",
        default_width="100%",
        default_height=f"{height}px",
    )


def _single_series_bar(data, x, y, title, labels, color_map=None):
    """Bar chart where each bar gets its own colour but the legend is hidden,
    since the x-axis already names every category — showing a legend too is
    redundant clutter for a one-bar-per-category chart."""
    fig = px.bar(data, x=x, y=y, title=title, labels=labels, color=x,
                 color_discrete_map=color_map)
    return _to_html(fig, hide_legend=True)


# ---------------------------------------------------------------- Halaman Utama
def home_charts(df: pd.DataFrame):
    charts = {}
    charts["distribusi_harga"] = _to_html(
        px.histogram(df, x=TARGET, nbins=40, title="Distribusi Harga Jual (Data Historis)")
    )
    avg_by_city = df.groupby("kota")[TARGET].mean().sort_values(ascending=False).reset_index()
    charts["harga_per_kota"] = _single_series_bar(
        avg_by_city, "kota", TARGET, "Rata-rata Harga per Kota", {TARGET: "Harga rata-rata (juta Rp)"}
    )
    avg_by_type = df.groupby("tipe_properti")[TARGET].mean().sort_values(ascending=False).reset_index()
    charts["harga_per_tipe"] = _single_series_bar(
        avg_by_type, "tipe_properti", TARGET, "Rata-rata Harga per Tipe Properti", {TARGET: "Harga rata-rata (juta Rp)"}
    )
    count_by_city = df["kota"].value_counts().reset_index()
    count_by_city.columns = ["kota", "jumlah"]
    fig = px.pie(count_by_city, names="kota", values="jumlah", title="Proporsi Properti per Kota", hole=0.62)
    fig.update_traces(textinfo="percent", textfont_size=11)
    fig.add_annotation(text=f"{len(df):,}<br><span style='font-size:10px;color:#6B7280'>properti</span>",
                        showarrow=False, font=dict(size=18, color="#111827", family="Inter"))
    charts["proporsi_kota"] = _to_html(fig)
    return charts


# ---------------------------------------------------------------- Visualisasi
def visualisasi_charts(df: pd.DataFrame):
    charts = {}

    fig = px.box(df, x="tipe_properti", y=TARGET, color="tipe_properti", title="Harga Jual per Tipe Properti")
    charts["harga_per_tipe_box"] = _to_html(fig, hide_legend=True)
    fig = px.box(df, x="kondisi_bangunan", y=TARGET, color="kondisi_bangunan",
                 category_orders={"kondisi_bangunan": ORDINAL_ORDER[0]}, color_discrete_map=KONDISI_COLORS,
                 title="Harga Jual per Kondisi Bangunan")
    charts["harga_per_kondisi_box"] = _to_html(fig, hide_legend=True)

    fig = px.scatter(df, x="luas_bangunan", y=TARGET, color="kota",
                      hover_data=["luas_tanah", "kamar_tidur", "usia_bangunan"],
                      title="Luas Bangunan vs Harga Jual (diwarnai per Kota)", opacity=0.65)
    fig.update_traces(marker=dict(size=6))
    charts["luas_vs_harga"] = _to_html(fig, height=420)

    num_cols = NUMERIC_FEATURES + ["garasi", TARGET]
    corr = df[num_cols].corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale=["#DC2626", "#FFFFFF", GREEN_DARK],
                     zmin=-1, zmax=1, title="Heatmap Korelasi", aspect="auto")
    fig.update_xaxes(showgrid=False, title="")
    fig.update_yaxes(showgrid=False, title="")
    fig.update_layout(margin=dict(l=140))
    charts["korelasi"] = _to_html(fig, height=460)

    harga_m2_kota = df.groupby("kota")["harga_per_m2"].mean().sort_values(ascending=False).reset_index()
    charts["harga_m2_kota"] = _single_series_bar(
        harga_m2_kota, "kota", "harga_per_m2", "Harga per m² menurut Kota", {"harga_per_m2": "Harga per m² (juta Rp)"}
    )
    harga_m2_tipe = df.groupby("tipe_properti")["harga_per_m2"].mean().sort_values(ascending=False).reset_index()
    charts["harga_m2_tipe"] = _single_series_bar(
        harga_m2_tipe, "tipe_properti", "harga_per_m2", "Harga per m² menurut Tipe Properti",
        {"harga_per_m2": "Harga per m² (juta Rp)"}
    )

    pivot_harga = df.pivot_table(index="kota", columns="tipe_properti", values=TARGET, aggfunc="mean")
    fig = px.imshow(pivot_harga, text_auto=".0f", color_continuous_scale=["#F0FDF4", GREEN_SOFT, GREEN_DARK],
                     title="Rata-rata Harga Jual (juta Rp) — Kota x Tipe Properti", labels={"color": "Harga (juta Rp)"},
                     aspect="auto")
    fig.update_xaxes(showgrid=False, title="")
    fig.update_yaxes(showgrid=False, title="")
    fig.update_layout(margin=dict(l=100))
    charts["peta_harga"] = _to_html(fig, height=400)

    jarak_bin = pd.cut(df["jarak_pusat_kota"], bins=[0, 10, 20, 30],
                        labels=["0-10 km", "10-20 km", "20-30 km"], include_lowest=True)
    harga_jarak = df.groupby(jarak_bin, observed=True)[TARGET].mean().reset_index()
    charts["harga_vs_jarak"] = _single_series_bar(
        harga_jarak, "jarak_pusat_kota", TARGET, "Harga Rata-rata vs Jarak ke Pusat Kota",
        {TARGET: "Harga rata-rata (juta Rp)", "jarak_pusat_kota": "Jarak"}
    )

    usia_bin = pd.cut(df["usia_bangunan"], bins=[0, 10, 20, 30, 40],
                       labels=["0-10 th", "10-20 th", "20-30 th", "30-40 th"], include_lowest=True)
    harga_usia = df.groupby(usia_bin, observed=True)[TARGET].mean().reset_index()
    fig = px.bar(harga_usia, x="usia_bangunan", y=TARGET, title="Harga Rata-rata vs Usia Bangunan",
                 labels={TARGET: "Harga rata-rata (juta Rp)", "usia_bangunan": "Usia bangunan"},
                 color="usia_bangunan", color_discrete_sequence=COLORWAY)
    fig.update_yaxes(range=[1700, 2000])
    charts["harga_vs_usia"] = _to_html(fig, hide_legend=True)

    harga_kt = df.groupby("kamar_tidur")["harga_per_m2"].mean().reset_index()
    fig = px.bar(harga_kt, x="kamar_tidur", y="harga_per_m2", title="Harga per m² vs Jumlah Kamar Tidur",
                 labels={"harga_per_m2": "Harga per m² (juta Rp)"}, color="kamar_tidur",
                 color_continuous_scale=["#F0FDF4", GREEN_SOFT, GREEN_DARK])
    charts["harga_m2_kamar_tidur"] = _to_html(fig, hide_legend=True)

    harga_kondisi = df.dropna(subset=["kondisi_bangunan"]).groupby("kondisi_bangunan")["harga_per_m2"].mean()
    harga_kondisi = harga_kondisi.reindex(ORDINAL_ORDER[0]).reset_index()
    fig = px.bar(harga_kondisi, x="kondisi_bangunan", y="harga_per_m2", title="Harga per m² vs Kondisi Bangunan",
                 labels={"harga_per_m2": "Harga per m² (juta Rp)"}, color="kondisi_bangunan",
                 color_discrete_map=KONDISI_COLORS)
    fig.update_yaxes(range=[6, 7])
    charts["harga_m2_kondisi"] = _to_html(fig, hide_legend=True)

    df_seg = df.copy()
    df_seg["segmen_harga"] = pd.qcut(df_seg[TARGET], 3, labels=["Murah", "Menengah", "Mewah"])
    komposisi_kota = (pd.crosstab(df_seg["segmen_harga"], df_seg["kota"], normalize="index") * 100).reset_index()
    charts["segmen_kota"] = _to_html(
        px.bar(komposisi_kota, x="segmen_harga", y=komposisi_kota.columns[1:],
               title="Komposisi Kota per Segmen Harga (%)", labels={"value": "Proporsi (%)"})
    )
    komposisi_tipe = (pd.crosstab(df_seg["segmen_harga"], df_seg["tipe_properti"], normalize="index") * 100).reset_index()
    charts["segmen_tipe"] = _to_html(
        px.bar(komposisi_tipe, x="segmen_harga", y=komposisi_tipe.columns[1:],
               title="Komposisi Tipe Properti per Segmen Harga (%)", labels={"value": "Proporsi (%)"})
    )

    return charts


# ---------------------------------------------------------------- Evaluasi Model
def evaluasi_charts(metrics_df: pd.DataFrame, test_pred_df: pd.DataFrame, best_model_name: str):
    # R2 is rendered as HTML progress bars in evaluasi.html instead of a chart,
    # to match the dashboard's progress-bar comparison style.
    charts = {}
    charts["rmse_bar"] = _single_series_bar(
        metrics_df.reset_index(), "index", "RMSE", "Perbandingan RMSE", {"index": "Model"}
    )
    fig = px.scatter(test_pred_df, x="harga_aktual", y="harga_prediksi", opacity=0.55,
                      labels={"harga_aktual": "Harga Aktual (juta Rp)", "harga_prediksi": "Harga Prediksi (juta Rp)"},
                      title=f"Aktual vs Prediksi — {best_model_name}",
                      color_discrete_sequence=[GREEN_MID])
    fig.update_traces(marker=dict(size=6))
    min_v, max_v = test_pred_df["harga_aktual"].min(), test_pred_df["harga_aktual"].max()
    fig.add_shape(type="line", x0=min_v, y0=min_v, x1=max_v, y1=max_v, line=dict(color="#9CA3AF", dash="dash", width=1.5))
    charts["aktual_vs_prediksi"] = _to_html(fig, height=420)
    return charts


# ---------------------------------------------------------------- Insight
def insight_charts(df: pd.DataFrame, importance_df: pd.DataFrame, best_model_name: str):
    charts = {}
    top_features = importance_df.head(5).copy()
    # Raw names carry the ColumnTransformer prefix (nom__/num__/ord__/bin__),
    # not fit for end-user display.
    top_features["feature"] = top_features["feature"].str.split("__", n=1).str[-1].str.replace("_", " ")
    fig = px.bar(top_features, x="importance", y="feature", orientation="h",
                 title=f"Top 5 Fitur Paling Berpengaruh ({best_model_name})", color="importance",
                 color_continuous_scale=["#F0FDF4", GREEN_SOFT, GREEN_DARK], labels={"feature": ""})
    fig.update_yaxes(categoryorder="total ascending")
    fig.update_layout(margin=dict(l=130))
    charts["top_features"] = _to_html(fig, hide_legend=True)

    harga_m2_kota = df.groupby("kota")["harga_per_m2"].mean().sort_values(ascending=False).reset_index()
    charts["harga_m2_kota"] = _single_series_bar(
        harga_m2_kota, "kota", "harga_per_m2", "Harga per m² menurut Kota", {"harga_per_m2": "Harga per m² (juta Rp)"}
    )

    df_seg = df.copy()
    df_seg["segmen_harga"] = pd.qcut(df_seg[TARGET], 3, labels=["Murah", "Menengah", "Mewah"])
    komposisi_kota_mewah = pd.crosstab(df_seg["segmen_harga"], df_seg["kota"], normalize="index").reset_index()
    charts["segmen_kota"] = _to_html(
        px.bar(komposisi_kota_mewah, x="segmen_harga", y=komposisi_kota_mewah.columns[1:],
               title="Komposisi Kota per Segmen Harga", labels={"value": "Proporsi"})
    )
    return charts


def batch_result_chart(result: pd.DataFrame):
    fig = px.histogram(result, x="harga_prediksi", nbins=30, title="Distribusi Hasil Prediksi (Agregat)")
    return _to_html(fig)
