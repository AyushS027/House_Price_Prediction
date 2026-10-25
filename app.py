import io
import re
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

st.set_page_config(page_title="HomeLens | Data Dashboard", page_icon="🏡", layout="wide")

SAMPLE = "house_prices_cleaned.csv"
SAMPLE_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
NONE = "— No target (explore only) —"
MUTE, LINE, TEAL, RUST, GOLD = "#8a948f", "#c9c2b6", "#2f6f73", "#c8553d", "#d9a441"
PALETTE = [TEAL, RUST, GOLD, "#8a5a9c", "#5b8c5a", "#3d6fb6", "#b5651d", "#7a7a7a"]

# ---- core start (no Streamlit calls) --------------------------------------------------
def fmt(v):
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def read_any_csv(raw=None):
    """Read any CSV: sniff separator, try several encodings, turn '$1,200' / '12%' into numbers, parse dates."""
    df = None
    if raw is None:
        df = pd.read_csv(SAMPLE)
    else:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", encoding=enc)
                break
            except Exception:
                df = None
        if df is None:
            raise ValueError("The file could not be read as a CSV")
    df.columns = [str(c).strip() or f"column_{i}" for i, c in enumerate(df.columns)]
    df = df.loc[:, ~pd.Index(df.columns).str.lower().str.startswith("unnamed")]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError("The file has no data rows")
    dupes = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            s = df[c].dropna().astype(str)
            if s.empty:
                continue
            n = pd.to_numeric(df[c].astype(str).str.replace(r"[,\s$€£₹%]", "", regex=True), errors="coerce")
            if n.notna().sum() / len(s) > .9:
                df[c] = n
            elif re.match(r"^\d{4}-\d{1,2}-\d{1,2}|^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", s.iloc[0]) or "date" in c.lower():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    dt = pd.to_datetime(df[c], errors="coerce")
                    if dt.notna().sum() / len(s) <= .9:
                        dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
                if dt.notna().sum() / len(s) > .9:
                    df[c] = dt
    return df, dupes


def kind_of(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_bool_dtype(s):
        return "category"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "category" if s.nunique(dropna=True) <= 30 else "text"


def task_of(s):
    n = s.nunique(dropna=True)
    if kind_of(s) == "numeric":
        return "regression" if n > 15 else ("classification" if n >= 2 else None)
    return "classification" if kind_of(s) == "category" and n >= 2 else None


def guess_target(df, options):
    keys = ("price", "value", "cost", "amount", "sale", "rent", "salary", "income", "revenue", "target",
            "label", "class", "outcome", "survived", "status", "species", "category")
    for k in keys:
        for c in options:
            if k in c.lower():
                return c
    return options[-1]


def candidate_features(df, target, task):
    t = pd.to_numeric(df[target], errors="coerce") if task == "regression" else None
    out = []
    for c in df.columns:
        s, n = df[c], df[c].nunique(dropna=True)
        k = kind_of(s)
        if c == target or n < 2 or s.isna().mean() > .5 or k in ("date", "text"):
            continue
        if re.search(r"(^|[_ ])id$", c.lower()) or (task == "regression" and target.lower() in c.lower()):
            continue
        if k == "numeric":
            if n == len(df) and pd.api.types.is_integer_dtype(s) and s.is_monotonic_increasing:
                continue
            if t is not None and abs(s.corr(t)) > .999:
                continue
        if t is not None and (k == "category" or n <= 2):
            grp = t.groupby(s).transform("mean")
            if 1 - ((t - grp) ** 2).sum() / max(((t - t.mean()) ** 2).sum(), 1e-12) > .9:
                continue
        out.append(c)
    return out


def prepare(df, target, feats, task):
    d = df[feats + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    if task == "classification":
        d[target] = d[target].astype(str)
        d = d[d[target].map(d[target].value_counts()) >= 2]
    if len(d) > 60000:
        d = d.sample(60000, random_state=42)
    num, cat = [], []
    for c in feats:
        if kind_of(d[c]) == "numeric":
            d[c] = d[c].fillna(d[c].median())
            num.append(c)
        else:
            fill = d[c].mode().iloc[0] if d[c].notna().any() else "missing"
            d[c] = d[c].fillna(fill).astype(str)
            cat.append(c)
    if len(d) < 30:
        raise ValueError("At least 30 usable rows are needed to train the models")
    return d, num, cat


def encode(X, cat, cols=None):
    E = pd.get_dummies(X, columns=cat, dtype=float)
    return E if cols is None else E.reindex(columns=cols, fill_value=0.0)


def train_all(d, target, num, cat, task):
    X, y = encode(d[num + cat], cat), d[target]
    if task == "regression":
        models = {"Linear Regression": LinearRegression(),
                  "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
                  "Decision Tree": DecisionTreeRegressor(max_depth=12, random_state=42),
                  "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)}
    else:
        models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                  "K-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsClassifier()),
                  "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
                  "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)}
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42,
                                                  stratify=y if task == "classification" else None)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
    rows, preds = [], {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[name] = p
        if task == "regression":
            rows.append({"Model": name, "MAE": mean_absolute_error(y_te, p),
                         "RMSE": np.sqrt(mean_squared_error(y_te, p)), "R²": r2_score(y_te, p)})
        else:
            rows.append({"Model": name, "Accuracy": accuracy_score(y_te, p),
                         "Precision": precision_score(y_te, p, average="weighted", zero_division=0),
                         "Recall": recall_score(y_te, p, average="weighted", zero_division=0),
                         "F1": f1_score(y_te, p, average="weighted", zero_division=0)})
    key, asc = ("RMSE", True) if task == "regression" else ("F1", False)
    metrics = pd.DataFrame(rows).sort_values(key, ascending=asc).reset_index(drop=True)
    return models, metrics, preds, y_te, list(X.columns)


def anova_scores(d, target, num):
    groups = [g.index for _, g in d.groupby(target) if len(g) > 1]
    out = {}
    for c in num:
        gs = [d.loc[i, c].values for i in groups]
        if len(gs) > 1 and d[c].std() > 0:
            out[c] = stats.f_oneway(*gs)
    return out
# ---- core end -------------------------------------------------------------------------

load_data = st.cache_data(read_any_csv)
build_models = st.cache_resource(train_all)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.block-container { padding-top:2rem; max-width:1250px; }
[data-testid="stSidebar"] { background:#1d2a2e; }
[data-testid="stSidebar"] * { color:#f1ebe0 !important; }
.eyebrow { color:#c8553d; text-transform:uppercase; letter-spacing:.14em; font-size:12px; font-weight:700; }
.title { font-family:Fraunces,Georgia,serif; font-size:42px; line-height:1.08; font-weight:700; margin:6px 0 8px; color:var(--text-color); }
.lead { color:var(--text-color); opacity:.65; max-width:640px; margin-bottom:12px; }
h3.sec { font-family:Fraunces,Georgia,serif; font-size:21px; margin:26px 0 4px; color:var(--text-color); }
.note { color:var(--text-color); opacity:.6; font-size:13px; margin-bottom:12px; }
.kpi { background:var(--secondary-background-color); border:1px solid rgba(128,128,128,.25); border-radius:18px; padding:18px 20px; min-height:128px; }
.kpi.hero { background:#1d2a2e; border-color:#1d2a2e; }
.kpi .l { font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--text-color); opacity:.6; }
.kpi .v { font-family:Fraunces,Georgia,serif; font-size:26px; font-weight:700; margin:6px 0 2px; color:var(--text-color); overflow-wrap:anywhere; }
.kpi .n { font-size:13px; color:var(--text-color); opacity:.6; }
.kpi.hero .l, .kpi.hero .n { color:#c9d3cf; opacity:1; } .kpi.hero .v { color:#f6f1ea; }
.result { background:#1d2a2e; border-radius:20px; padding:28px; text-align:center; color:#f6f1ea; }
.result .big { font-family:Fraunces,Georgia,serif; font-size:48px; font-weight:700; color:#ee7a5f; overflow-wrap:anywhere; }
div[data-testid="stFormSubmitButton"]>button { background:#c8553d; color:white; border:0; border-radius:12px; font-weight:700; }
footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def kpi(label, value, note="", hero=False):
    return f'<div class="kpi {"hero" if hero else ""}"><div class="l">{label}</div><div class="v">{value}</div><div class="n">{note}</div></div>'


def head(title, note=""):
    st.markdown(f'<h3 class="sec">{title}</h3><div class="note">{note}</div>', unsafe_allow_html=True)


def style(ax, xl="", yl=""):
    ax.figure.patch.set_alpha(0)
    ax.set_facecolor("none")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTE, labelsize=9)
    ax.set_xlabel(xl, color=MUTE, fontsize=10)
    ax.set_ylabel(yl, color=MUTE, fontsize=10)
    ax.grid(axis="y", color=LINE, lw=.5, alpha=.6)
    ax.set_axisbelow(True)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def hbar(series, color=TEAL, xl=""):
    fig, ax = plt.subplots(figsize=(6, max(2.6, .3 * len(series) + 1)))
    ax.barh([str(i) for i in series.index], series.values, color=color)
    style(ax, xl)
    ax.grid(axis="x", color=LINE, lw=.5, alpha=.6)
    show(fig)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🏡 HomeLens")
    st.caption("Insights for any CSV file")
    nav = st.container()
    st.markdown("---")
    st.markdown("**Your data**")
    upload = st.file_uploader("Upload any CSV", type=["csv", "txt", "tsv"], help="Any table works. Numbers, text and dates are detected automatically.")
    st.download_button("Download sample template", pd.DataFrame(columns=SAMPLE_FEATURES + ["Price"]).to_csv(index=False), "template.csv", "text/csv")
    setup = st.container()
    status = st.empty()

raw = upload.getvalue() if upload else None
source = f"Uploaded: {upload.name}" if upload else "Sample dataset"
target = task = None
try:
    try:
        df, dupes = load_data(raw)
    except Exception as e:
        if raw is None:
            raise
        st.warning(f"{e}. Showing the sample dataset instead.")
        raw, source = None, "Sample dataset"
        df, dupes = load_data(None)
    kinds = {c: kind_of(df[c]) for c in df.columns}
    opts = [c for c in df.columns if kinds[c] != "date" and task_of(df[c])]
    with setup:
        if opts:
            choices = [NONE] + opts
            t_def = "Price" if raw is None and "Price" in opts else guess_target(df, opts)
            tsel = st.selectbox("Column to predict", choices, index=choices.index(t_def), key=f"t_{source}",
                                help="Numbers give a regression, categories give a classification.")
            target = None if tsel == NONE else tsel
        else:
            st.caption("No column looks suitable as a prediction target, so profiling and exploring are available.")
        if target:
            task = task_of(df[target])
            pool = candidate_features(df, target, task)
            f_def = SAMPLE_FEATURES if raw is None and target == "Price" else pool
            feats = st.multiselect("Feature columns", [c for c in df.columns if c != target and kinds[c] != "date"],
                                   default=[c for c in f_def if c != target], key=f"f_{source}_{target}")
            if not feats:
                st.warning("Select at least one feature column.")
                target = None
    if target:
        d, num, cat = prepare(df, target, feats, task)
        models, metrics, preds, y_test, enc_cols = build_models(d, target, num, cat, task)
except Exception as e:
    st.error(f"Could not prepare the data: {e}")
    st.stop()

pages = ["Data profile"] + (["Overview", "Model performance", "Predictor", "Statistics"] if target else []) + ["Explore data"]
with nav:
    page = st.radio("Navigate", pages, label_visibility="collapsed")
status.caption(f"{source} · {len(df):,} rows · {len(df.columns)} columns" + (f" · {task}" if task else ""))

reg = task == "regression"
if target:
    best = metrics.iloc[0]
    classes = sorted(d[target].unique()) if not reg else []
    tmean = d[target].mean() if reg else None
    corr = d[num].corrwith(d[target]).dropna().sort_values() if reg and num else pd.Series(dtype=float)
    top = corr.abs().idxmax() if len(corr) else None
    if not reg:
        an = anova_scores(d, target, num)
        top = max(an, key=lambda k: an[k][0]) if an else None
    lat = next((c for c in df.columns if c.lower() in ("latitude", "lat")), None)
    lon = next((c for c in df.columns if c.lower() in ("longitude", "lon", "lng", "long")), None)

title = f"What drives {target}?" if target else "Get to know your data"
st.markdown(f'<div class="eyebrow">{"Regression" if reg else "Classification" if target else "Data profile"}</div>'
            f'<div class="title">{title}</div>'
            '<div class="lead">Upload any CSV and the dashboard adapts: it profiles every column, trains four models when a target is chosen, and explains the result.</div>',
            unsafe_allow_html=True)
st.caption(f"📂 Data source: {source}")

# -------------------------------------------------------------- Data profile
if page == "Data profile":
    nn = sum(k == "numeric" for k in kinds.values())
    nc = sum(k == "category" for k in kinds.values())
    c = st.columns(4)
    c[0].markdown(kpi("Rows", f"{len(df):,}", f"{dupes:,} duplicates removed", True), unsafe_allow_html=True)
    c[1].markdown(kpi("Columns", len(df.columns), f"{nn} numeric · {nc} categorical"), unsafe_allow_html=True)
    c[2].markdown(kpi("Missing cells", f"{df.isna().mean().mean() * 100:.1f}%", f"{int(df.isna().sum().sum()):,} empty values"), unsafe_allow_html=True)
    c[3].markdown(kpi("Date / text", f"{sum(k == 'date' for k in kinds.values())} / {sum(k == 'text' for k in kinds.values())}", "columns"), unsafe_allow_html=True)

    head("Columns", "Detected type, completeness and an example value for every column.")
    schema = pd.DataFrame({"Column": df.columns, "Type": [kinds[c] for c in df.columns],
                           "Missing %": [round(df[c].isna().mean() * 100, 1) for c in df.columns],
                           "Unique": [df[c].nunique() for c in df.columns],
                           "Example": [str(df[c].dropna().iloc[0])[:30] if df[c].notna().any() else "" for c in df.columns]})
    st.dataframe(schema, hide_index=True, use_container_width=True)

    head("Inspect a column")
    col = st.selectbox("Column", list(df.columns), label_visibility="collapsed")
    s = df[col].dropna()
    a, b = st.columns(2)
    with a:
        if kinds[col] == "numeric":
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.hist(s, bins=min(40, max(5, s.nunique())), color=TEAL)
            style(ax, col, "Rows")
            show(fig)
        elif kinds[col] == "date":
            m = s.dt.to_period("M").value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.plot(m.index.to_timestamp(), m.values, color=TEAL)
            style(ax, col, "Rows per month")
            show(fig)
        else:
            hbar(s.astype(str).value_counts().head(15).sort_values(), TEAL, "Rows")
    with b:
        if kinds[col] == "numeric":
            st.dataframe(s.describe().round(3).to_frame("value"), use_container_width=True)
            st.caption(f"Skewness {s.skew():.2f} · Kurtosis {s.kurt():.2f}")
        else:
            st.dataframe(s.astype(str).value_counts().head(15).rename("Rows").to_frame(), use_container_width=True)
    ncols = [c for c in df.columns if kinds[c] == "numeric"]
    if len(ncols) >= 2:
        head("Correlation heatmap", "How numeric columns move together (top 15 by variance).")
        use = df[ncols].var().sort_values(ascending=False).head(15).index
        cm = df[use].corr()
        fig, ax = plt.subplots(figsize=(8, 5.4))
        im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(use)), use, rotation=45, ha="right", fontsize=8, color=MUTE)
        ax.set_yticks(range(len(use)), use, fontsize=8, color=MUTE)
        if len(use) <= 10:
            for i in range(len(use)):
                for j in range(len(use)):
                    ax.text(j, i, f"{cm.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#222")
        fig.patch.set_alpha(0)
        fig.colorbar(im, ax=ax)
        show(fig)

# ----------------------------------------------------------------- Overview
elif page == "Overview":
    c = st.columns(4)
    if reg:
        c[0].markdown(kpi(f"Average {target}", fmt(tmean), f"Median {fmt(d[target].median())}", True), unsafe_allow_html=True)
        c[1].markdown(kpi("Best model", best.Model, f"R² = {best['R²']:.3f}"), unsafe_allow_html=True)
        c[2].markdown(kpi("Best RMSE", fmt(best.RMSE), f"MAE {fmt(best.MAE)}"), unsafe_allow_html=True)
        c[3].markdown(kpi("Top driver", top or "n/a", f"Correlation {corr[top]:+.2f}" if top else "No numeric features"), unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            head(f"{target} distribution", "Dashed line marks the median.")
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.hist(d[target], bins=40, color=TEAL)
            ax.axvline(d[target].median(), color=RUST, ls="--", lw=1.5)
            style(ax, target, "Rows")
            show(fig)
        with b:
            head(f"Correlation with {target}", "Pearson r for each numeric feature.")
            if len(corr):
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.barh(corr.index, corr.values, color=[RUST if v > 0 else "#8a5a9c" for v in corr.values])
                style(ax, "Correlation")
                ax.grid(axis="x", color=LINE, lw=.5, alpha=.6)
                show(fig)
            else:
                st.info("Add a numeric feature to see correlations.")
        a, b = st.columns(2)
        with a:
            if top:
                head(f"{target} by {top} level", "Average across the three terciles.")
                g = d.groupby(pd.qcut(d[top], 3, duplicates="drop"), observed=True)[target].mean()
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.bar([f"T{i + 1}" for i in range(len(g))], g.values, color=[GOLD, TEAL, RUST][:len(g)], width=.55)
                style(ax, f"{top} tercile", f"Average {target}")
                show(fig)
        with b:
            if lat and lon and kinds[lat] == "numeric" and kinds[lon] == "numeric":
                head("Map view", f"Each dot is a row, coloured by {target}.")
                g = df.loc[d.index].assign(_t=d[target]).sample(min(4000, len(d)), random_state=42)
                fig, ax = plt.subplots(figsize=(6, 3.4))
                sc = ax.scatter(g[lon], g[lat], c=g["_t"], s=6, cmap="YlOrRd", alpha=.7)
                style(ax, "Longitude", "Latitude")
                fig.colorbar(sc, ax=ax, label=target)
                show(fig)
            elif top:
                head(f"{top} vs {target}", "Each dot is a row.")
                s = d.sample(min(3000, len(d)), random_state=42)
                fig, ax = plt.subplots(figsize=(6, 3.4))
                ax.scatter(s[top], s[target], s=7, alpha=.35, color=TEAL)
                style(ax, top, target)
                show(fig)
    else:
        vc = d[target].value_counts()
        c[0].markdown(kpi("Classes", len(classes), f"Largest: {vc.index[0]} ({vc.iloc[0] / len(d):.0%})", True), unsafe_allow_html=True)
        c[1].markdown(kpi("Best model", best.Model, f"F1 = {best.F1:.3f}"), unsafe_allow_html=True)
        c[2].markdown(kpi("Accuracy", f"{best.Accuracy:.1%}", f"Precision {best.Precision:.1%}"), unsafe_allow_html=True)
        c[3].markdown(kpi("Top separating feature", top or "n/a", f"ANOVA F = {an[top][0]:,.0f}" if top else "No numeric features"), unsafe_allow_html=True)
        a, b = st.columns(2)
        with a:
            head(f"Class balance of {target}", "Rows per class.")
            hbar(vc.head(15).sort_values(), TEAL, "Rows")
        with b:
            if top:
                head(f"{top} by {target}", "Distribution of the most informative numeric feature per class.")
                cls = list(vc.head(8).index)
                fig, ax = plt.subplots(figsize=(6, 3.6))
                bp = ax.boxplot([d.loc[d[target] == k, top] for k in cls], labels=cls, patch_artist=True, showfliers=False)
                for p, col in zip(bp["boxes"], PALETTE):
                    p.set_facecolor(col)
                    p.set_alpha(.7)
                style(ax, target, top)
                show(fig)
            else:
                st.info("Add a numeric feature to compare classes.")

# -------------------------------------------------------- Model performance
elif page == "Model performance":
    head("Model leaderboard", "Four models trained on an 80/20 split. " + ("Lower RMSE is better." if reg else "Higher F1 is better."))
    st.dataframe(metrics.style.format({k: ("{:,.2f}" if reg and k != "R²" else "{:.3f}") for k in metrics.columns[1:]}), hide_index=True, use_container_width=True)
    mk = ("RMSE", "R²") if reg else ("Accuracy", "F1")
    a, b = st.columns(2)
    for col, m in zip((a, b), mk):
        with col:
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.bar(metrics.Model, metrics[m], color=[RUST] + [TEAL] * 3, width=.55)
            ax.set_title(m + (" (lower is better)" if m == "RMSE" else " (higher is better)"), color=MUTE, fontsize=11, loc="left")
            ax.tick_params(axis="x", rotation=15)
            style(ax)
            show(fig)
    pick = st.selectbox("Inspect a model", metrics.Model.tolist())
    a, b = st.columns(2)
    with a:
        if reg:
            head("Actual vs predicted", "Dots near the dashed line are accurate.")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(y_test, preds[pick], s=6, alpha=.35, color=TEAL)
            lim = [min(y_test.min(), preds[pick].min()), max(y_test.max(), preds[pick].max())]
            ax.plot(lim, lim, "--", color=RUST)
            style(ax, f"Actual {target}", f"Predicted {target}")
        else:
            head("Confusion matrix", "Rows are actual classes, columns are predictions.")
            cls = list(classes)[:12]
            cm = confusion_matrix(y_test, preds[pick], labels=cls)
            fig, ax = plt.subplots(figsize=(6, 4.2))
            ax.imshow(cm, cmap="YlGnBu")
            ax.set_xticks(range(len(cls)), cls, rotation=45, ha="right", fontsize=8, color=MUTE)
            ax.set_yticks(range(len(cls)), cls, fontsize=8, color=MUTE)
            for i in range(len(cls)):
                for j in range(len(cls)):
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="#222")
            ax.set_xlabel("Predicted", color=MUTE)
            ax.set_ylabel("Actual", color=MUTE)
            fig.patch.set_alpha(0)
        show(fig)
    with b:
        head("What the model relies on", "Feature importance or scaled coefficient size.")
        m = models[pick]
        scaled = hasattr(m, "steps")
        core = m[-1] if scaled else m
        imp = None
        if hasattr(core, "feature_importances_"):
            imp = core.feature_importances_
        elif hasattr(core, "coef_"):
            imp = np.abs(np.atleast_2d(core.coef_)).mean(0)
            if not scaled:
                imp = imp * encode(d[num + cat], cat, enc_cols).std().values
        if imp is None:
            st.info("This model type has no built-in importance scores. Try Random Forest or Decision Tree.")
        else:
            hbar(pd.Series(imp / max(imp.sum(), 1e-12), index=enc_cols).sort_values().tail(12), GOLD, "Relative importance")

# ---------------------------------------------------------------- Predictor
elif page == "Predictor":
    head(f"Predict {target}", "Adjust the inputs, pick a model and get an instant result.")
    with st.form("predict"):
        model_name = st.selectbox("Model", metrics.Model.tolist())
        cols = st.columns(4)
        vals = {}
        for i, f in enumerate(feats):
            if f in num:
                lo, hi, med = float(d[f].min()), float(d[f].max()), float(d[f].median())
                vals[f] = cols[i % 4].number_input(f, lo, hi, med, step=(hi - lo) / 100 or 1.0)
            else:
                vals[f] = cols[i % 4].selectbox(f, sorted(d[f].unique()))
        go = st.form_submit_button("Predict")
    if go:
        row = encode(pd.DataFrame([vals]), cat, enc_cols)
        m = models[model_name]
        if reg:
            rmse = float(metrics.set_index("Model").loc[model_name, "RMSE"])
            st.markdown(f'<div class="result"><div>Estimated {target} · {model_name}</div><div class="big">{fmt(float(m.predict(row)[0]))}</div>'
                        f'<div>Typical error ± {fmt(rmse)}</div></div>', unsafe_allow_html=True)
        else:
            pr = pd.Series(m.predict_proba(row)[0], index=m.classes_ if hasattr(m, "classes_") else m[-1].classes_).sort_values(ascending=False)
            st.markdown(f'<div class="result"><div>Predicted {target} · {model_name}</div><div class="big">{pr.index[0]}</div>'
                        f'<div>Confidence {pr.iloc[0]:.1%}</div></div>', unsafe_allow_html=True)
            hbar(pr.head(8).sort_values(), RUST, "Probability")

# --------------------------------------------------------------- Statistics
elif page == "Statistics":
    head("Statistical tests", "Computed live from your data. Significant means p < 0.05.")
    rows = []
    if reg:
        ref = st.number_input(f"Reference mean for the t-test ({target})", value=float(round(tmean)), step=max(float(d[target].std()) / 20, 1e-6), format="%.2f")
        n = min(500, len(d))
        sample = d[target].sample(n, random_state=42)
        t, p = stats.ttest_1samp(sample, ref)
        lo, hi = stats.t.interval(.95, n - 1, loc=sample.mean(), scale=stats.sem(sample))
        rows.append((f"{target}", "One-sample t-test", f"t = {t:.3f}", p))
        if top:
            ps = d.sample(min(5000, len(d)), random_state=42)
            pr, pp = stats.pearsonr(ps[top], ps[target])
            sr, sp = stats.spearmanr(ps[top], ps[target])
            rows += [(top, "Pearson correlation", f"r = {pr:.3f}", pp), (top, "Spearman correlation", f"ρ = {sr:.3f}", sp)]
            gs = [g[target].values for _, g in d.groupby(pd.qcut(d[top], 3, labels=False, duplicates="drop")) if len(g) > 1]
            if len(gs) > 1:
                F, fp = stats.f_oneway(*gs)
                rows.append((top, "ANOVA across terciles", f"F = {F:,.1f}", fp))
    else:
        counts = d[target].value_counts().values
        chi, cp = stats.chisquare(counts)
        rows.append((target, "Chi-square (equal classes)", f"χ² = {chi:,.1f}", cp))
        for c in num[:15]:
            if c in an:
                rows.append((c, "ANOVA across classes", f"F = {an[c][0]:,.1f}", an[c][1]))
        for c in cat[:10]:
            ct = pd.crosstab(d[c], d[target])
            if ct.shape[0] > 1 and ct.shape[1] > 1:
                chi, cp, _, _ = stats.chi2_contingency(ct)
                rows.append((c, "Chi-square independence", f"χ² = {chi:,.1f}", cp))
    out = pd.DataFrame(rows, columns=["Variable", "Test", "Statistic", "p-value"])
    out["Result"] = np.where(out["p-value"] < .05, "Significant", "Not significant")
    out["p-value"] = out["p-value"].map(lambda v: "< 0.001" if v < .001 else f"{v:.3f}")
    st.dataframe(out, hide_index=True, use_container_width=True)
    if reg:
        st.info(f"95% confidence interval for the mean {target} (n = {n}): {fmt(lo)} to {fmt(hi)}")
        head("Central Limit Theorem", "1,000 resamples of n = 100 produce a bell-shaped distribution of means.")
        rng = np.random.default_rng(42)
        means = [rng.choice(d[target].values, 100).mean() for _ in range(1000)]
        fig, ax = plt.subplots(figsize=(8, 3.3))
        ax.hist(means, bins=30, color=TEAL)
        ax.axvline(tmean, color=RUST, ls="--")
        style(ax, f"Sample mean of {target}", "Count")
        show(fig)
        c = st.columns(2)
        c[0].metric("Theoretical std. error", fmt(d[target].std() / 10))
        c[1].metric("Observed std. error", fmt(np.std(means, ddof=1)))
    else:
        st.caption("ANOVA tests whether a numeric feature differs between classes. Chi-square tests whether a categorical feature is linked to the class.")

# ------------------------------------------------------------- Explore data
else:
    head("Explore the dataset", "Filter and download what you see.")
    view = df
    if target and reg:
        lo, hi = float(d[target].min()), float(d[target].max())
        r = st.slider(f"{target} range", lo, hi, (lo, hi), step=(hi - lo) / 100 or 1.0)
        view = df[pd.to_numeric(df[target], errors="coerce").between(*r)]
    elif target:
        keep = st.multiselect(f"{target} values", classes, default=classes)
        view = df[df[target].astype(str).isin(keep)]
    st.caption(f"{len(view):,} of {len(df):,} rows")
    st.dataframe(view, use_container_width=True, height=380)
    st.download_button("Download filtered CSV", view.to_csv(index=False), "filtered_data.csv", "text/csv")
    head("Summary statistics")
    st.dataframe(view.describe(include="all").T.astype(str), use_container_width=True)
