"""Rumahku House Price — Aplikasi Sains Data (FastAPI).

Jalankan dengan: uvicorn app.main:app --reload
(dari folder root proyek, C:\\DASD\\tugasbesardasd)
"""
import base64
import io
import os

import pandas as pd
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import charts
from .services import (
    ALL_FEATURES, ORDINAL_ORDER, TARGET, get_valid_categories, load_data,
    load_feature_importance, load_metrics, load_model, load_test_predictions,
    validate_and_clean_batch,
)

APP_DIR = os.path.dirname(__file__)
app = FastAPI(title="Rumahku House Price")
app.mount("/static", StaticFiles(directory=os.path.join(APP_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(APP_DIR, "templates"))

NAV_ITEMS = [
    {"path": "/", "icon": "bi-house", "label": "Halaman Utama"},
    {"path": "/dataset", "icon": "bi-folder2-open", "label": "Tampilan Dataset"},
    {"path": "/visualisasi", "icon": "bi-bar-chart", "label": "Visualisasi"},
    {"path": "/prediksi", "icon": "bi-robot", "label": "Prediksi Harga"},
    {"path": "/evaluasi", "icon": "bi-graph-up", "label": "Evaluasi Model"},
    {"path": "/insight", "icon": "bi-lightbulb", "label": "Insight"},
    {"path": "/rekomendasi", "icon": "bi-clipboard-check", "label": "Rekomendasi"},
    {"path": "/dokumentasi", "icon": "bi-book", "label": "Dokumentasi"},
]


def base_context(request: Request, active_page: str) -> dict:
    _, best_name = load_model()
    _, df = load_data()
    page_title = next((item["label"] for item in NAV_ITEMS if item["path"] == active_page), "")
    return {
        "request": request,
        "nav_items": NAV_ITEMS,
        "active_page": active_page,
        "page_title": page_title,
        "best_model_name": best_name,
        "n_data": f"{len(df):,}",
    }


# ================================================================== Halaman Utama
@app.get("/", response_class=HTMLResponse)
def halaman_utama(request: Request):
    _, df = load_data()
    ctx = base_context(request, "/")
    ctx.update({
        "jumlah_properti": f"{len(df):,}",
        "harga_rata2": f"Rp {df[TARGET].mean():,.0f} jt",
        "harga_median": f"Rp {df[TARGET].median():,.0f} jt",
        "charts": charts.home_charts(df),
    })
    return templates.TemplateResponse(request, "home.html", ctx)


# ================================================================== Tampilan Dataset
@app.get("/dataset", response_class=HTMLResponse)
def tampilan_dataset(request: Request, n: int = 10):
    df_raw, df = load_data()
    n = max(5, min(n, 100))

    missing = df_raw.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    ctx = base_context(request, "/dataset")
    ctx.update({
        "n_rows_shown": n,
        "sample_table": df.head(n).to_html(classes="table table-sm table-striped dataframe", index=False, border=0, justify="left"),
        "describe_table": df.describe(include="all").T.to_html(classes="table table-sm table-striped dataframe", border=0, justify="left"),
        "missing_table": missing.rename("jumlah_missing").to_frame().to_html(
            classes="table table-sm table-striped dataframe", border=0, justify="left") if len(missing) else None,
        "n_duplicated": int(df_raw.duplicated().sum()),
        "n_dropped_target": int(df_raw.shape[0] - df.shape[0]),
        "shape_clean": f"{df.shape[0]} baris x {df.shape[1]} kolom",
        "shape_raw": f"{df_raw.shape[0]} baris",
    })
    return templates.TemplateResponse(request, "dataset.html", ctx)


# ================================================================== Visualisasi
@app.get("/visualisasi", response_class=HTMLResponse)
def visualisasi(request: Request):
    _, df = load_data()
    ctx = base_context(request, "/visualisasi")
    ctx["charts"] = charts.visualisasi_charts(df)
    return templates.TemplateResponse(request, "visualisasi.html", ctx)


# ================================================================== Prediksi
@app.get("/prediksi", response_class=HTMLResponse)
def prediksi_form(request: Request):
    _, df = load_data()
    valid_kota, valid_tipe = get_valid_categories(df)
    ctx = base_context(request, "/prediksi")
    ctx.update({
        "valid_kota": valid_kota,
        "valid_tipe": valid_tipe,
        "kondisi_options": ORDINAL_ORDER[0],
        "all_features": ALL_FEATURES,
        "prediction": None,
        "batch_result": None,
    })
    return templates.TemplateResponse(request, "prediksi.html", ctx)


@app.post("/prediksi/manual", response_class=HTMLResponse)
def prediksi_manual(
    request: Request,
    luas_tanah: float = Form(...),
    luas_bangunan: float = Form(...),
    kamar_tidur: int = Form(...),
    kamar_mandi: int = Form(...),
    jumlah_lantai: int = Form(...),
    usia_bangunan: float = Form(...),
    jarak_pusat_kota: float = Form(...),
    kondisi_bangunan: str = Form(...),
    tipe_properti: str = Form(...),
    kota: str = Form(...),
    garasi: str = Form(...),
):
    model, best_name = load_model()
    metrics_df = load_metrics()
    _, df = load_data()
    valid_kota, valid_tipe = get_valid_categories(df)

    input_df = pd.DataFrame([{
        "luas_tanah": luas_tanah,
        "luas_bangunan": luas_bangunan,
        "kamar_tidur": kamar_tidur,
        "kamar_mandi": kamar_mandi,
        "jumlah_lantai": jumlah_lantai,
        "usia_bangunan": usia_bangunan,
        "jarak_pusat_kota": jarak_pusat_kota,
        "kondisi_bangunan": kondisi_bangunan,
        "tipe_properti": tipe_properti,
        "garasi": 1 if garasi == "Ya" else 0,
        "kota": kota,
    }])

    prediction = None
    error = None
    try:
        pred = model.predict(input_df)[0]
        mae = metrics_df.loc[best_name, "MAE"]
        prediction = {
            "harga": f"Rp {pred:,.1f} juta",
            "rentang": f"Rp {pred - mae:,.0f} juta – Rp {pred + mae:,.0f} juta",
        }
    except Exception as e:
        error = f"Gagal membuat prediksi: {e}"

    ctx = base_context(request, "/prediksi")
    ctx.update({
        "valid_kota": valid_kota,
        "valid_tipe": valid_tipe,
        "kondisi_options": ORDINAL_ORDER[0],
        "all_features": ALL_FEATURES,
        "prediction": prediction,
        "error": error,
        "batch_result": None,
    })
    return templates.TemplateResponse(request, "prediksi.html", ctx)


@app.post("/prediksi/batch", response_class=HTMLResponse)
async def prediksi_batch(request: Request, file: UploadFile):
    model, best_name = load_model()
    _, df = load_data()
    valid_kota, valid_tipe = get_valid_categories(df)

    batch_result = None
    error = None
    warnings_list = []

    raw_bytes = await file.read()
    if not raw_bytes:
        error = "File kosong — tidak ada isi yang diunggah."
    else:
        try:
            df_upload = pd.read_csv(io.BytesIO(raw_bytes))
        except Exception as e:
            error = f"File tidak bisa dibaca sebagai CSV yang valid: {e}"
        else:
            clean, warnings_list, fatal_errors = validate_and_clean_batch(df_upload, df)
            if fatal_errors:
                error = " ".join(fatal_errors)
            else:
                try:
                    preds = model.predict(clean)
                except Exception as e:
                    error = f"Model gagal memproses data yang diunggah: {e}"
                else:
                    result = df_upload.head(len(clean)).copy()
                    result["harga_prediksi"] = preds
                    csv_b64 = base64.b64encode(result.to_csv(index=False).encode("utf-8")).decode()
                    batch_result = {
                        "n_rows": len(result),
                        "table": result.to_html(classes="table table-sm table-striped dataframe", index=False, border=0, justify="left"),
                        "chart": charts.batch_result_chart(result),
                        "download_href": f"data:text/csv;base64,{csv_b64}",
                    }

    ctx = base_context(request, "/prediksi")
    ctx.update({
        "valid_kota": valid_kota,
        "valid_tipe": valid_tipe,
        "kondisi_options": ORDINAL_ORDER[0],
        "all_features": ALL_FEATURES,
        "prediction": None,
        "error": error,
        "warnings_list": warnings_list,
        "batch_result": batch_result,
    })
    return templates.TemplateResponse(request, "prediksi.html", ctx)


@app.get("/prediksi/template.csv")
def download_template():
    template_df = pd.DataFrame([{c: "" for c in ALL_FEATURES}])
    buf = io.StringIO()
    template_df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=template_prediksi_rumahku.csv"},
    )


# ================================================================== Evaluasi Model
@app.get("/evaluasi", response_class=HTMLResponse)
def evaluasi(request: Request):
    metrics_df = load_metrics()
    test_pred_df = load_test_predictions()
    _, best_name = load_model()
    _, df = load_data()

    model_bars = [
        {"name": name, "r2_pct": round(row["R2"] * 100, 1), "is_best": name == best_name}
        for name, row in metrics_df.sort_values("R2", ascending=False).iterrows()
    ]

    ctx = base_context(request, "/evaluasi")
    ctx.update({
        "metrics_table": metrics_df.round(3).to_html(classes="table table-sm table-striped dataframe", border=0, justify="left"),
        "charts": charts.evaluasi_charts(metrics_df, test_pred_df, best_name),
        "model_bars": model_bars,
        "rmse_pct": f"{metrics_df.loc[best_name, 'RMSE'] / df[TARGET].mean() * 100:.1f}",
        "r2_pct": f"{metrics_df.loc[best_name, 'R2']*100:.1f}",
        "rmse_val": f"{metrics_df.loc[best_name, 'RMSE']:,.0f}",
        "r2_val": f"{metrics_df.loc[best_name, 'R2']:.3f}",
        "harga_rata2": f"{df[TARGET].mean():,.0f}",
    })
    return templates.TemplateResponse(request, "evaluasi.html", ctx)


# ================================================================== Insight
@app.get("/insight", response_class=HTMLResponse)
def insight(request: Request):
    _, df = load_data()
    importance_df = load_feature_importance()
    metrics_df = load_metrics()
    _, best_name = load_model()

    harga_m2_kota = df.groupby("kota")["harga_per_m2"].mean().sort_values(ascending=False)
    harga_m2_tipe = df.groupby("tipe_properti")["harga_per_m2"].mean().sort_values(ascending=False)
    pivot_harga = df.pivot_table(index="kota", columns="tipe_properti", values=TARGET, aggfunc="mean")
    termahal_kota, termahal_tipe = pivot_harga.stack().idxmax()
    termurah_kota, termurah_tipe = pivot_harga.stack().idxmin()

    jarak_bin = pd.cut(df["jarak_pusat_kota"], bins=[0, 10, 20, 30],
                        labels=["0-10 km", "10-20 km", "20-30 km"], include_lowest=True)
    harga_jarak = df.groupby(jarak_bin, observed=True)[TARGET].mean()
    penurunan_jarak = (harga_jarak.iloc[0] - harga_jarak.iloc[-1]) / harga_jarak.iloc[0] * 100

    df_seg = df.copy()
    df_seg["segmen_harga"] = pd.qcut(df_seg[TARGET], 3, labels=["Murah", "Menengah", "Mewah"])
    pct_mewah_jakarta = pd.crosstab(df_seg["segmen_harga"], df_seg["kota"], normalize="index").loc["Mewah", "Jakarta"] * 100
    pct_mewah_villa = pd.crosstab(df_seg["segmen_harga"], df_seg["tipe_properti"], normalize="index").loc["Mewah", "Villa"] * 100

    total_importance = importance_df["importance"].sum()
    top_feat_label = importance_df.iloc[0]["feature"].split("__", 1)[-1].replace("_", " ")
    top_feat_share = importance_df.iloc[0]["importance"] / total_importance * 100
    luas_share = (importance_df[importance_df["feature"].str.contains("luas_")]["importance"].sum()
                  / total_importance * 100)
    tipe_premium_pct = (harga_m2_tipe.iloc[0] / harga_m2_tipe.iloc[-1] - 1) * 100

    ctx = base_context(request, "/insight")
    ctx.update({
        "charts": charts.insight_charts(df, importance_df, best_name),
        "top_feat_label": top_feat_label,
        "top_feat_share": f"{top_feat_share:.0f}",
        "luas_share": f"{luas_share:.0f}",
        "kota_termahal": harga_m2_kota.index[0],
        "harga_m2_termahal": f"{harga_m2_kota.iloc[0]:.1f}",
        "kota_termurah": harga_m2_kota.index[-1],
        "harga_m2_termurah": f"{harga_m2_kota.iloc[-1]:.1f}",
        "tipe_termahal": harga_m2_tipe.index[0],
        "tipe_termurah": harga_m2_tipe.index[-1],
        "tipe_premium_pct": f"{tipe_premium_pct:.0f}",
        "termahal_tipe": termahal_tipe,
        "termahal_kota": termahal_kota,
        "termurah_tipe": termurah_tipe,
        "termurah_kota": termurah_kota,
        "segmen_ratio": f"{pivot_harga.loc[termahal_kota, termahal_tipe] / pivot_harga.loc[termurah_kota, termurah_tipe]:.1f}",
        "harga_termahal_segmen": f"{pivot_harga.loc[termahal_kota, termahal_tipe]:,.0f}",
        "penurunan_jarak": f"{penurunan_jarak:.0f}",
        "pct_mewah_jakarta": f"{pct_mewah_jakarta:.0f}",
        "pct_mewah_villa": f"{pct_mewah_villa:.0f}",
        "best_r2": f"{metrics_df.loc[best_name, 'R2']:.3f}",
        "linreg_r2": f"{metrics_df.loc['Linear Regression', 'R2']:.3f}",
    })
    return templates.TemplateResponse(request, "insight.html", ctx)


# ================================================================== Rekomendasi
@app.get("/rekomendasi", response_class=HTMLResponse)
def rekomendasi(request: Request):
    ctx = base_context(request, "/rekomendasi")
    return templates.TemplateResponse(request, "rekomendasi.html", ctx)


# ================================================================== Dokumentasi
@app.get("/dokumentasi", response_class=HTMLResponse)
def dokumentasi(request: Request):
    ctx = base_context(request, "/dokumentasi")
    return templates.TemplateResponse(request, "dokumentasi.html", ctx)
