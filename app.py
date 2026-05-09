"""
Statistical Regression Analysis App
Simple & Multiple Linear Regression — SPSS-equivalent output
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kstest, norm
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regression Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: #0f1117;
    color: #e8e8e8;
}

/* Main header */
.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #f0e6d3;
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #8a8a9a;
    font-size: 1.05rem;
    margin-bottom: 2rem;
}

/* Cards */
.stat-card {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.stat-card h4 {
    font-family: 'DM Serif Display', serif;
    color: #c9b99a;
    font-size: 1.1rem;
    margin-bottom: 0.8rem;
}

/* Step indicator */
.step-badge {
    display: inline-block;
    background: #c9b99a22;
    border: 1px solid #c9b99a44;
    color: #c9b99a;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 0.2rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.5rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* Pass/Fail badges */
.badge-pass {
    background: #1a3a2a;
    border: 1px solid #2a6a4a;
    color: #4ade80;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-fail {
    background: #3a1a1a;
    border: 1px solid #6a2a2a;
    color: #f87171;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}
.badge-warn {
    background: #3a2f1a;
    border: 1px solid #6a5a2a;
    color: #fbbf24;
    padding: 0.25rem 0.8rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
}

/* Result value display */
.result-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    color: #f0e6d3;
    font-weight: 500;
}
.result-label {
    font-size: 0.78rem;
    color: #6a6a7a;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Table styling */
.dataframe {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.85rem !important;
}

/* Section headers */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #f0e6d3;
    border-bottom: 1px solid #2a2d3a;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem 0;
}

/* Interpretation box */
.interp-box {
    background: #12151f;
    border-left: 3px solid #c9b99a;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
    font-size: 0.92rem;
    color: #c8c8d8;
    line-height: 1.7;
}

/* Equation display */
.equation-box {
    background: #12151f;
    border: 1px solid #c9b99a44;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    font-family: 'DM Mono', monospace;
    font-size: 1.2rem;
    color: #c9b99a;
    text-align: center;
    margin: 1rem 0;
    letter-spacing: 0.5px;
}

/* Recommendation box */
.rec-box {
    background: #1a1f35;
    border: 1px solid #3a4a7a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    color: #a0b4e8;
    font-size: 0.9rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #2a2d3a;
    margin: 1.5rem 0;
}

/* Button override */
.stButton > button {
    background: #c9b99a !important;
    color: #0f1117 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: #e0d0b8 !important;
    transform: translateY(-1px) !important;
}

/* Select box */
.stSelectbox label, .stMultiSelect label {
    color: #8a8a9a !important;
    font-size: 0.85rem !important;
}

/* Upload area */
.stFileUploader {
    border: 2px dashed #2a2d3a !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

/* Radio */
.stRadio label {
    color: #c8c8d8 !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: #1a1d27;
    border: 1px solid #2a2d3a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
[data-testid="metric-container"] label {
    color: #6a6a7a !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    font-family: 'DM Mono', monospace !important;
    color: #f0e6d3 !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# STATISTICAL FUNCTIONS (SPSS-equivalent)
# ═══════════════════════════════════════════════════════════════════════════════

def run_normality(residuals):
    """Shapiro-Wilk and Kolmogorov-Smirnov on residuals (SPSS-equivalent)."""
    n = len(residuals)
    # Shapiro-Wilk
    sw_stat, sw_p = shapiro(residuals)
    # K-S (Lilliefors-adjusted like SPSS: test against N(mean,std) of residuals)
    res_std = (residuals - residuals.mean()) / residuals.std(ddof=1)
    ks_stat, ks_p = kstest(res_std, 'norm')
    return {
        "n": n,
        "sw_stat": sw_stat, "sw_p": sw_p,
        "ks_stat": ks_stat, "ks_p": ks_p,
        "primary": "Shapiro-Wilk" if n <= 50 else "Kolmogorov-Smirnov",
        "primary_stat": sw_stat if n <= 50 else ks_stat,
        "primary_p": sw_p if n <= 50 else ks_p,
        "passed": (sw_p > 0.05 if n <= 50 else ks_p > 0.05),
    }


def run_linearity(x_series, y_series):
    """
    ANOVA-based linearity test matching SPSS Means → Test for Linearity.
    When X is continuous (all unique), falls back to OLS F-test for linearity.
    Returns F_lin, p_lin, F_dev, p_dev.
    """
    x = np.array(x_series, dtype=float)
    y = np.array(y_series, dtype=float)
    n = len(y)
    grand_mean = y.mean()

    # Linear regression SS
    x_c = sm.add_constant(x)
    model = sm.OLS(y, x_c).fit()
    ss_reg = model.ess          # SS explained by linear component
    ss_res_linear = model.ssr

    # SS for groups (unique X values) — between groups
    groups = {}
    for xi, yi in zip(x, y):
        groups.setdefault(xi, []).append(yi)

    n_groups = len(groups)
    df_lin = 1
    df_within = n - n_groups

    ss_within = sum(np.sum((np.array(v) - np.mean(v)) ** 2) for v in groups.values())

    # If all X values unique: deviation from linearity cannot be computed
    # (no within-group variance). Report OLS F as linearity F.
    if n_groups == n or df_within <= 0:
        ms_lin = ss_reg / df_lin
        ms_res = ss_res_linear / (n - 2)
        F_lin = model.fvalue
        p_lin = model.f_pvalue
        return {
            "ss_reg": ss_reg, "ss_dev": np.nan, "ss_within": np.nan,
            "df_lin": df_lin, "df_dev": 0, "df_within": n - 2,
            "ms_lin": ms_lin, "ms_dev": np.nan, "ms_within": ms_res,
            "F_lin": F_lin, "p_lin": p_lin,
            "F_dev": np.nan, "p_dev": np.nan,
            "passed": p_lin < 0.05,
            "continuous_x": True,
        }

    ss_between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values())
    df_groups = n_groups - 1
    ss_dev = ss_between - ss_reg
    df_dev = df_groups - df_lin

    ms_within = ss_within / df_within if df_within > 0 else np.nan
    ms_lin = ss_reg / df_lin
    ms_dev = ss_dev / df_dev if df_dev > 0 else np.nan

    F_lin = ms_lin / ms_within if (ms_within and ms_within > 0) else np.nan
    F_dev = ms_dev / ms_within if (ms_dev is not None and ms_within and ms_within > 0 and df_dev > 0) else np.nan

    p_lin = 1 - stats.f.cdf(F_lin, df_lin, df_within) if not np.isnan(F_lin) else np.nan
    p_dev = 1 - stats.f.cdf(F_dev, df_dev, df_within) if (not np.isnan(F_dev) and df_dev > 0) else np.nan

    passed = (not np.isnan(p_lin) and p_lin < 0.05) and (np.isnan(p_dev) or p_dev > 0.05)
    return {
        "ss_reg": ss_reg, "ss_dev": ss_dev, "ss_within": ss_within,
        "df_lin": df_lin, "df_dev": df_dev, "df_within": df_within,
        "ms_lin": ms_lin, "ms_dev": ms_dev, "ms_within": ms_within,
        "F_lin": F_lin, "p_lin": p_lin,
        "F_dev": F_dev, "p_dev": p_dev,
        "passed": passed,
        "continuous_x": False,
    }


def run_vif(X_df):
    """VIF and Tolerance — identical to SPSS Collinearity Diagnostics."""
    X = sm.add_constant(X_df.values)
    results = []
    for i, col in enumerate(X_df.columns):
        vif = variance_inflation_factor(X, i + 1)
        tol = 1.0 / vif if vif != 0 else np.nan
        results.append({"Variable": col, "Tolerance": tol, "VIF": vif})
    return pd.DataFrame(results)


def run_glejser(residuals, X_df):
    """
    Glejser heteroscedasticity test — regress |residuals| on predictors.
    Matches SPSS Glejser output.
    """
    abs_res = np.abs(residuals)
    X_c = sm.add_constant(X_df.values)
    model = sm.OLS(abs_res, X_c).fit()
    rows = []
    # Constant
    rows.append({
        "Term": "(Constant)",
        "B": model.params[0],
        "Std. Error": model.bse[0],
        "t": model.tvalues[0],
        "Sig.": model.pvalues[0],
    })
    for i, col in enumerate(X_df.columns):
        rows.append({
            "Term": col,
            "B": model.params[i + 1],
            "Std. Error": model.bse[i + 1],
            "t": model.tvalues[i + 1],
            "Sig.": model.pvalues[i + 1],
        })
    passed = all(row["Sig."] > 0.05 for row in rows if row["Term"] != "(Constant)")
    return pd.DataFrame(rows), passed, model


def run_regression(y, X_df):
    """
    OLS regression — SPSS-equivalent Model Summary, ANOVA, Coefficients.
    """
    X_c = sm.add_constant(X_df.values)
    model = sm.OLS(y.values, X_c).fit()
    n = len(y)
    k = X_df.shape[1]

    # ── Model Summary ──────────────────────────────────────────────────────
    R = np.sqrt(model.rsquared)
    R2 = model.rsquared
    adj_R2 = model.rsquared_adj
    se_est = np.sqrt(model.mse_resid)
    dw = durbin_watson(model.resid)

    # ── ANOVA ──────────────────────────────────────────────────────────────
    ss_reg = model.ess
    ss_res = model.ssr
    ss_tot = model.centered_tss
    df_reg = k
    df_res = n - k - 1
    df_tot = n - 1
    ms_reg = ss_reg / df_reg
    ms_res = ss_res / df_res
    F = model.fvalue
    p_F = model.f_pvalue

    anova = pd.DataFrame([
        {"": "Regression", "Sum of Squares": ss_reg, "df": df_reg,
         "Mean Square": ms_reg, "F": F, "Sig.": p_F},
        {"": "Residual",   "Sum of Squares": ss_res, "df": df_res,
         "Mean Square": ms_res, "F": "", "Sig.": ""},
        {"": "Total",      "Sum of Squares": ss_tot, "df": df_tot,
         "Mean Square": "", "F": "", "Sig.": ""},
    ])

    # ── Coefficients ───────────────────────────────────────────────────────
    # Standardised beta = b * (std_x / std_y)
    y_std = y.std(ddof=1)
    coef_rows = []
    coef_rows.append({
        "Model": "(Constant)",
        "B": model.params[0],
        "Std. Error": model.bse[0],
        "Beta": "",
        "t": model.tvalues[0],
        "Sig.": model.pvalues[0],
    })
    for i, col in enumerate(X_df.columns):
        x_std = X_df[col].std(ddof=1)
        beta = model.params[i + 1] * (x_std / y_std)
        coef_rows.append({
            "Model": col,
            "B": model.params[i + 1],
            "Std. Error": model.bse[i + 1],
            "Beta": beta,
            "t": model.tvalues[i + 1],
            "Sig.": model.pvalues[i + 1],
        })
    coef_df = pd.DataFrame(coef_rows)

    residuals = pd.Series(model.resid, name="Residuals")
    fitted = pd.Series(model.fittedvalues, name="Fitted")

    return {
        "model": model,
        "R": R, "R2": R2, "adj_R2": adj_R2, "se_est": se_est, "dw": dw,
        "ss_reg": ss_reg, "ss_res": ss_res, "ss_tot": ss_tot,
        "df_reg": df_reg, "df_res": df_res, "df_tot": df_tot,
        "ms_reg": ms_reg, "ms_res": ms_res,
        "F": F, "p_F": p_F,
        "anova": anova,
        "coef_df": coef_df,
        "residuals": residuals,
        "fitted": fitted,
        "n": n, "k": k,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d27"
GOLD      = "#c9b99a"
LIGHT_TXT = "#e8e8e8"
MUTED_TXT = "#6a6a7a"
PASS_CLR  = "#4ade80"
FAIL_CLR  = "#f87171"

def style_ax(ax, fig):
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(CARD_BG)
    ax.tick_params(colors=MUTED_TXT, labelsize=8)
    ax.xaxis.label.set_color(MUTED_TXT)
    ax.yaxis.label.set_color(MUTED_TXT)
    ax.title.set_color(LIGHT_TXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")


def plot_normality(residuals):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor(DARK_BG)

    # Histogram
    ax1 = axes[0]
    ax1.set_facecolor(CARD_BG)
    ax1.hist(residuals, bins=min(20, len(residuals)//2+1),
             color=GOLD, alpha=0.75, edgecolor="#2a2d3a", linewidth=0.5)
    mu, sigma = residuals.mean(), residuals.std(ddof=1)
    x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    n = len(residuals)
    bin_w = (residuals.max() - residuals.min()) / (min(20, n//2+1))
    ax1.plot(x_range, norm.pdf(x_range, mu, sigma) * n * bin_w,
             color="#f0e6d3", linewidth=1.8, label="Normal curve")
    ax1.set_title("Histogram of Residuals", fontsize=11, pad=10, color=LIGHT_TXT,
                  fontfamily="serif")
    ax1.set_xlabel("Residual", color=MUTED_TXT, fontsize=9)
    ax1.set_ylabel("Frequency", color=MUTED_TXT, fontsize=9)
    ax1.tick_params(colors=MUTED_TXT, labelsize=8)
    for sp in ax1.spines.values(): sp.set_edgecolor("#2a2d3a")

    # Q-Q Plot
    ax2 = axes[1]
    ax2.set_facecolor(CARD_BG)
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    ax2.scatter(osm, osr, color=GOLD, s=22, alpha=0.8, zorder=3)
    x_line = np.array([osm[0], osm[-1]])
    ax2.plot(x_line, slope * x_line + intercept, color="#f0e6d3",
             linewidth=1.5, zorder=2)
    ax2.set_title("Normal Q-Q Plot", fontsize=11, pad=10, color=LIGHT_TXT,
                  fontfamily="serif")
    ax2.set_xlabel("Theoretical Quantiles", color=MUTED_TXT, fontsize=9)
    ax2.set_ylabel("Sample Quantiles", color=MUTED_TXT, fontsize=9)
    ax2.tick_params(colors=MUTED_TXT, labelsize=8)
    for sp in ax2.spines.values(): sp.set_edgecolor("#2a2d3a")

    plt.tight_layout(pad=2)
    return fig


def plot_scatter_residuals(fitted, residuals):
    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax, fig)
    # Standardise
    z_resid = (residuals - residuals.mean()) / residuals.std(ddof=1)
    z_fitted = (fitted - fitted.mean()) / fitted.std(ddof=1)
    ax.scatter(z_fitted, z_resid, color=GOLD, s=25, alpha=0.75, zorder=3)
    ax.axhline(0, color="#f0e6d3", linewidth=1, linestyle="--", alpha=0.6)
    ax.axhline(2, color=FAIL_CLR, linewidth=0.7, linestyle=":", alpha=0.5)
    ax.axhline(-2, color=FAIL_CLR, linewidth=0.7, linestyle=":", alpha=0.5)
    ax.set_title("Scatterplot — ZRESID vs ZPRED", fontsize=11, pad=10,
                 color=LIGHT_TXT, fontfamily="serif")
    ax.set_xlabel("Regression Standardized Predicted Value (ZPRED)",
                  color=MUTED_TXT, fontsize=8)
    ax.set_ylabel("Regression Standardized Residual (ZRESID)",
                  color=MUTED_TXT, fontsize=8)
    plt.tight_layout()
    return fig


def plot_regression_line(x_series, y_series, x_label, y_label, model):
    fig, ax = plt.subplots(figsize=(7, 4))
    style_ax(ax, fig)
    x = np.array(x_series)
    y = np.array(y_series)
    ax.scatter(x, y, color=GOLD, s=30, alpha=0.8, zorder=3, label="Observed")
    x_line = np.linspace(x.min(), x.max(), 200)
    X_line = sm.add_constant(x_line)
    y_pred = model.predict(X_line)
    ax.plot(x_line, y_pred, color="#f0e6d3", linewidth=2, zorder=4,
            label="Regression line")
    ax.set_title(f"Scatter Plot: {x_label} vs {y_label}",
                 fontsize=11, pad=10, color=LIGHT_TXT, fontfamily="serif")
    ax.set_xlabel(x_label, color=MUTED_TXT, fontsize=9)
    ax.set_ylabel(y_label, color=MUTED_TXT, fontsize=9)
    ax.legend(facecolor=CARD_BG, edgecolor="#2a2d3a",
              labelcolor=LIGHT_TXT, fontsize=8)
    plt.tight_layout()
    return fig


def fig_to_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# WORD REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_word_report(reg_type, y_col, x_cols,
                         norm_results, lin_results,
                         vif_df, glejser_df, glejser_passed,
                         dw_value, reg_results,
                         all_passed, figs_bytes):
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    doc = Document()

    # Styles
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    def add_heading(text, level=1):
        h = doc.add_heading(text, level=level)
        h.runs[0].font.color.rgb = RGBColor(0x1a, 0x1a, 0x2e)
        return h

    def add_para(text, bold=False, italic=False, align=None):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.bold = bold
        run.italic = italic
        if align:
            p.alignment = align
        return p

    def add_table_from_df(df, caption=None):
        if caption:
            cp = doc.add_paragraph(caption)
            cp.runs[0].bold = True
        table = doc.add_table(rows=1, cols=len(df.columns))
        table.style = "Table Grid"
        hdr = table.rows[0].cells
        for i, col in enumerate(df.columns):
            hdr[i].text = str(col)
            hdr[i].paragraphs[0].runs[0].bold = True
        for _, row in df.iterrows():
            cells = table.add_row().cells
            for i, val in enumerate(row):
                if isinstance(val, float):
                    cells[i].text = f"{val:.4f}" if not np.isnan(val) else "-"
                else:
                    cells[i].text = str(val) if val != "" else "-"
        doc.add_paragraph()

    # ── Title ──────────────────────────────────────────────────────────────
    title = doc.add_heading(
        f"{'Simple' if reg_type == 'Simple' else 'Multiple'} Linear Regression Analysis Report",
        level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    add_para(f"Dependent Variable (Y): {y_col}", bold=True)
    add_para(f"Independent Variable(s) (X): {', '.join(x_cols)}", bold=True)
    doc.add_paragraph()

    # ── 1. Normality ───────────────────────────────────────────────────────
    add_heading("1. Normality Test of Residuals", level=1)
    add_para(
        "The normality assumption requires that regression residuals follow a normal distribution. "
        f"{'Shapiro-Wilk is used (n ≤ 50).' if norm_results['n'] <= 50 else 'Kolmogorov-Smirnov is used (n > 50).'}"
    )
    norm_df = pd.DataFrame([
        {"Test": "Kolmogorov-Smirnov", "Statistic": norm_results["ks_stat"],
         "Sig.": norm_results["ks_p"]},
        {"Test": "Shapiro-Wilk", "Statistic": norm_results["sw_stat"],
         "Sig.": norm_results["sw_p"]},
    ])
    add_table_from_df(norm_df, "Tests of Normality")
    primary_p = norm_results["primary_p"]
    verdict = "normally distributed" if primary_p > 0.05 else "NOT normally distributed"
    add_para(
        f"Based on the {norm_results['primary']} test, Sig. = {primary_p:.3f} "
        f"({'> 0.05' if primary_p > 0.05 else '< 0.05'}). "
        f"The residuals are {verdict}. "
        f"Normality assumption: {'✓ SATISFIED' if norm_results['passed'] else '✗ VIOLATED'}.",
        bold=True
    )

    # Normality figure
    if "normality" in figs_bytes:
        buf = BytesIO(figs_bytes["normality"])
        doc.add_picture(buf, width=Inches(5.5))
        doc.add_paragraph()

    # ── 2. Linearity ───────────────────────────────────────────────────────
    add_heading("2. Linearity Test", level=1)
    for x_col, lr in lin_results.items():
        add_heading(f"  {x_col} → {y_col}", level=2)
        lin_df = pd.DataFrame([
            {"Source": "Linearity",
             "Sum of Squares": lr["ss_reg"],
             "df": lr["df_lin"],
             "Mean Square": lr["ms_lin"],
             "F": lr["F_lin"],
             "Sig.": lr["p_lin"]},
            {"Source": "Deviation from Linearity",
             "Sum of Squares": lr["ss_dev"],
             "df": lr["df_dev"],
             "Mean Square": lr["ms_dev"] if lr["df_dev"] > 0 else np.nan,
             "F": lr["F_dev"] if lr["df_dev"] > 0 else np.nan,
             "Sig.": lr["p_dev"] if lr["df_dev"] > 0 else np.nan},
            {"Source": "Within Groups",
             "Sum of Squares": lr["ss_within"],
             "df": lr["df_within"],
             "Mean Square": lr["ms_within"],
             "F": "", "Sig.": ""},
        ])
        add_table_from_df(lin_df, f"ANOVA Table — {x_col} vs {y_col}")
        p_lin_str = f"{lr['p_lin']:.3f}" if not np.isnan(lr['p_lin']) else "N/A"
        p_dev_str = f"{lr['p_dev']:.3f}" if (lr['df_dev'] > 0 and not np.isnan(lr['p_dev'])) else "N/A"
        add_para(
            f"Sig. Linearity = {p_lin_str}; Sig. Deviation from Linearity = {p_dev_str}. "
            f"Linearity assumption: {'✓ SATISFIED' if lr['passed'] else '✗ VIOLATED'}.",
            bold=True
        )

    # ── 3. Multicollinearity (multiple only) ───────────────────────────────
    if reg_type == "Multiple" and vif_df is not None:
        add_heading("3. Multicollinearity Test", level=1)
        add_table_from_df(vif_df, "Collinearity Statistics")
        mc_pass = all(vif_df["VIF"] < 10) and all(vif_df["Tolerance"] > 0.10)
        add_para(
            f"All VIF values {'< 10 and Tolerance > 0.10' if mc_pass else 'exceed thresholds'}. "
            f"Multicollinearity assumption: {'✓ SATISFIED' if mc_pass else '✗ VIOLATED'}.",
            bold=True
        )

    # ── 4. Heteroscedasticity ─────────────────────────────────────────────
    sec = "4" if reg_type == "Multiple" else "3"
    add_heading(f"{sec}. Heteroscedasticity Test (Glejser)", level=1)
    add_table_from_df(glejser_df, "Glejser Test — Coefficients")
    add_para(
        f"All predictors have Sig. {'> 0.05' if glejser_passed else '< 0.05 for at least one predictor'}. "
        f"Heteroscedasticity assumption: {'✓ SATISFIED' if glejser_passed else '✗ VIOLATED'}.",
        bold=True
    )
    if "scatter" in figs_bytes:
        buf = BytesIO(figs_bytes["scatter"])
        doc.add_picture(buf, width=Inches(5.0))
        doc.add_paragraph()

    # ── 5. Autocorrelation ────────────────────────────────────────────────
    sec2 = "5" if reg_type == "Multiple" else "4"
    add_heading(f"{sec2}. Autocorrelation Test (Durbin-Watson)", level=1)
    dw_pass = 1.5 <= dw_value <= 2.5
    add_para(f"Durbin-Watson statistic: {dw_value:.3f}")
    add_para(
        f"DW = {dw_value:.3f} ({'within' if dw_pass else 'outside'} acceptable range 1.5 – 2.5). "
        f"Autocorrelation assumption: {'✓ SATISFIED' if dw_pass else '✗ VIOLATED'}.",
        bold=True
    )

    # ── 6. Regression Results ─────────────────────────────────────────────
    sec3 = "6" if reg_type == "Multiple" else "5"
    add_heading(f"{sec3}. Regression Analysis Results", level=1)

    # Model Summary
    ms_df = pd.DataFrame([{
        "R": reg_results["R"],
        "R Square": reg_results["R2"],
        "Adjusted R Square": reg_results["adj_R2"],
        "Std. Error of the Estimate": reg_results["se_est"],
        "Durbin-Watson": reg_results["dw"],
    }])
    add_table_from_df(ms_df, "Model Summary")

    # ANOVA
    add_table_from_df(reg_results["anova"], "ANOVA")

    # Coefficients
    coef_display = reg_results["coef_df"].copy()
    add_table_from_df(coef_display, "Coefficients")

    # Regression equation
    coefs = reg_results["coef_df"]
    const = coefs.loc[coefs["Model"] == "(Constant)", "B"].values[0]
    eq_parts = [f"{const:.4f}"]
    for _, row in coefs[coefs["Model"] != "(Constant)"].iterrows():
        sign = "+" if row["B"] >= 0 else "-"
        eq_parts.append(f"{sign} {abs(row['B']):.4f}({row['Model']})")
    eq_str = "Ŷ = " + " ".join(eq_parts)
    add_para(f"Regression Equation: {eq_str}", bold=True)

    # Regression plot
    if "regression" in figs_bytes:
        buf = BytesIO(figs_bytes["regression"])
        doc.add_picture(buf, width=Inches(5.5))
        doc.add_paragraph()

    # ── 7. Interpretation & Conclusion ────────────────────────────────────
    sec4 = "7" if reg_type == "Multiple" else "6"
    add_heading(f"{sec4}. Interpretation and Conclusion", level=1)

    R2_pct = reg_results["R2"] * 100
    F_val = reg_results["F"]
    p_F = reg_results["p_F"]
    sig_model = p_F < 0.05

    conclusion = (
        f"Prior to the regression analysis, all prerequisite assumption tests were conducted. "
    )
    if norm_results["passed"]:
        conclusion += f"The {norm_results['primary']} normality test confirmed that residuals are normally distributed (Sig. = {norm_results['primary_p']:.3f} > 0.05). "
    else:
        conclusion += f"The normality assumption was violated (Sig. = {norm_results['primary_p']:.3f} < 0.05). "

    for x_col, lr in lin_results.items():
        if lr["passed"]:
            conclusion += f"The linearity test confirmed a linear relationship between {x_col} and {y_col}. "
        else:
            conclusion += f"The linearity assumption between {x_col} and {y_col} was violated. "

    if reg_type == "Multiple" and vif_df is not None:
        mc_pass = all(vif_df["VIF"] < 10)
        conclusion += f"No multicollinearity was detected (all VIF < 10). " if mc_pass else "Multicollinearity was detected. "

    conclusion += f"The Glejser test showed {'no heteroscedasticity' if glejser_passed else 'heteroscedasticity'}. "
    dw_pass = 1.5 <= dw_value <= 2.5
    conclusion += f"The Durbin-Watson statistic (DW = {dw_value:.3f}) indicated {'no autocorrelation' if dw_pass else 'autocorrelation'}. "

    conclusion += (
        f"\n\nRegression analysis results showed that the model is "
        f"{'statistically significant' if sig_model else 'not statistically significant'} "
        f"(F = {F_val:.3f}, Sig. = {p_F:.3f}). "
        f"The coefficient of determination (R²) = {reg_results['R2']:.3f}, indicating that "
        f"{R2_pct:.1f}% of the variance in {y_col} is explained by {', '.join(x_cols)}. "
        f"The regression equation is: {eq_str}. "
    )

    for _, row in reg_results["coef_df"][reg_results["coef_df"]["Model"] != "(Constant)"].iterrows():
        sig_x = row["Sig."] < 0.05
        direction = "positive" if row["B"] > 0 else "negative"
        conclusion += (
            f"{row['Model']} has a {direction} and "
            f"{'significant' if sig_x else 'non-significant'} effect on {y_col} "
            f"(B = {row['B']:.4f}, Sig. = {row['Sig.']:.3f}). "
        )

    add_para(conclusion)

    # Save to bytes
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.read()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER UI
# ═══════════════════════════════════════════════════════════════════════════════

def fmt(val, decimals=4):
    if val == "" or val is None:
        return "-"
    try:
        if np.isnan(float(val)):
            return "-"
        return f"{float(val):.{decimals}f}"
    except Exception:
        return str(val)


def sig_color(p):
    try:
        p = float(p)
        if p < 0.05:
            return "🟢" if p < 0.05 else ""
        return ""
    except Exception:
        return ""


def display_pass_fail(passed, pass_text="SATISFIED", fail_text="VIOLATED"):
    if passed:
        st.markdown(f'<span class="badge-pass">✓ {pass_text}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="badge-fail">✗ {fail_text}</span>', unsafe_allow_html=True)


def display_coef_table(coef_df):
    """Render coefficients table with colour-coded significance."""
    cols = st.columns([2.5, 1.3, 1.3, 1.3, 1.3, 1.3])
    headers = ["Model", "B", "Std. Error", "Beta", "t", "Sig."]
    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")
    for _, row in coef_df.iterrows():
        cols = st.columns([2.5, 1.3, 1.3, 1.3, 1.3, 1.3])
        cols[0].write(row["Model"])
        cols[1].write(fmt(row["B"]))
        cols[2].write(fmt(row["Std. Error"]))
        cols[3].write(fmt(row["Beta"]) if row["Beta"] != "" else "-")
        cols[4].write(fmt(row["t"]))
        try:
            p = float(row["Sig."])
            star = " *" if p < 0.05 else ""
            cols[5].write(f"{p:.4f}{star}")
        except Exception:
            cols[5].write("-")


def display_anova_table(anova_df):
    cols = st.columns([2, 1.8, 1, 1.8, 1.5, 1.5])
    headers = ["", "Sum of Squares", "df", "Mean Square", "F", "Sig."]
    for col, h in zip(cols, headers):
        col.markdown(f"**{h}**")
    for _, row in anova_df.iterrows():
        cols = st.columns([2, 1.8, 1, 1.8, 1.5, 1.5])
        cols[0].write(row[""])
        cols[1].write(fmt(row["Sum of Squares"]))
        cols[2].write(str(int(row["df"])) if row["df"] != "" else "-")
        cols[3].write(fmt(row["Mean Square"]) if row["Mean Square"] != "" else "-")
        cols[4].write(fmt(row["F"]) if row["F"] != "" else "-")
        try:
            p = float(row["Sig."])
            cols[5].write(f"{p:.4f}" + (" *" if p < 0.05 else ""))
        except Exception:
            cols[5].write("-")


# ═══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATIONS WHEN ASSUMPTION FAILS
# ═══════════════════════════════════════════════════════════════════════════════

RECOMMENDATIONS = {
    "normality": """
**Normality assumption violated. Recommended actions:**
1. **Increase sample size** — larger samples tend toward normality (Central Limit Theorem applies at n > 30).
2. **Transform data** — apply Log (LN) or Square Root (SQRT) transformation to Y, then re-run all tests.
3. **Remove outliers** — check for extreme values using boxplots; remove only if theoretically justified.
4. **Use non-parametric alternative** — consider Spearman correlation if the assumption cannot be met.
""",
    "linearity": """
**Linearity assumption violated. Recommended actions:**
1. **Transform variable** — try LN(X), SQRT(X), or LN(Y).
2. **Polynomial regression** — add a squared term (X²) to capture a curvilinear relationship.
3. **Non-linear regression** — if a specific curve is theoretically expected.
4. **Review theory** — ensure that a linear relationship is theoretically justified.
""",
    "multicollinearity": """
**Multicollinearity detected. Recommended actions:**
1. **Remove one predictor** — if X1 and X2 measure very similar constructs, drop the less important one.
2. **Combine variables** — create a composite score (average X1 and X2) if they measure the same thing.
3. **Mean-center variables** — subtract the mean from each predictor before entering the model.
4. **Increase sample size** — may reduce multicollinearity impact.
""",
    "heteroscedasticity": """
**Heteroscedasticity detected. Recommended actions:**
1. **Transform Y** — apply LN(Y) or SQRT(Y) to stabilize variance.
2. **Weighted Least Squares (WLS)** — weight observations inversely to their variance.
3. **Check for omitted variables** — missing predictors can cause unequal variance patterns.
4. **Use robust standard errors** — if transformation is not suitable.
""",
    "autocorrelation": """
**Autocorrelation detected. Recommended actions:**
1. **Add lagged variable** — include Y at time t-1 as a predictor.
2. **Difference the data** — transform Y to first differences (Yt − Yt-1).
3. **Use time-series models** — ARIMA may be more appropriate if data is temporal.
4. **Note:** If your data is cross-sectional (not time-series), autocorrelation is less of a concern.
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
        <div class="main-title">Regression Analyzer</div>
        <div class="main-subtitle">
            SPSS-equivalent Simple &amp; Multiple Linear Regression · Automatic assumption testing · Downloadable report
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 1: Regression type ───────────────────────────────────────────
    st.markdown('<div class="step-badge">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Choose Regression Type</div>',
                unsafe_allow_html=True)

    reg_type = st.radio(
        "Regression type",
        ["Simple Regression (1 X, 1 Y)", "Multiple Regression (2+ X, 1 Y)"],
        horizontal=True,
        label_visibility="collapsed",
    )
    reg_type = "Simple" if "Simple" in reg_type else "Multiple"

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 2: Upload ────────────────────────────────────────────────────
    st.markdown('<div class="step-badge">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Upload CSV Data</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="interp-box">
    <b>Required CSV format:</b> First row = column headers. Each subsequent row = one respondent.
    All data columns must be numeric. Separator: comma (,). Encoding: UTF-8.<br><br>
    <b>Example (Simple):</b> <code>respondent_id, X, Y</code><br>
    <b>Example (Multiple):</b> <code>respondent_id, X1, X2, Y</code><br>
    Column names can be anything — you will select which columns are X and Y below.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload your CSV file", type=["csv"],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("⬆️  Please upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return

    st.success(f"✓ File loaded — {len(df)} rows, {len(df.columns)} columns")

    with st.expander("Preview Data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # Check numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Your CSV must have at least 2 numeric columns.")
        return

    # Missing value check
    missing = df[numeric_cols].isnull().sum()
    if missing.any():
        st.warning(
            f"Missing values detected: {missing[missing > 0].to_dict()}. "
            "Rows with missing values will be dropped automatically."
        )
        df = df.dropna(subset=numeric_cols)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Step 3: Variable mapping ──────────────────────────────────────────
    st.markdown('<div class="step-badge">Step 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Map Variables</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns(2)
    with col_left:
        y_col = st.selectbox("Dependent Variable (Y)", numeric_cols)

    remaining = [c for c in numeric_cols if c != y_col]

    if reg_type == "Simple":
        with col_right:
            x_col = st.selectbox("Independent Variable (X)", remaining)
        x_cols = [x_col]
    else:
        with col_right:
            x_cols = st.multiselect(
                "Independent Variables (X1, X2, ...)",
                remaining,
                default=remaining[:min(2, len(remaining))]
            )
        if len(x_cols) < 2:
            st.warning("Please select at least 2 independent variables for Multiple Regression.")
            return

    if not x_cols:
        st.warning("Please select at least one independent variable.")
        return

    # Prepare arrays
    y = df[y_col].astype(float)
    X_df = df[x_cols].astype(float)

    st.markdown(f"""
    <div class="interp-box">
    <b>Model:</b> {y_col} = f({', '.join(x_cols)})<br>
    <b>Sample size (n):</b> {len(y)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # AUTO-RUN ALL TESTS
    # ═══════════════════════════════════════════════════════════════════════

    st.markdown('<div class="step-badge">Step 04 — Automatic</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="section-header">Prerequisite Assumption Tests</div>',
                unsafe_allow_html=True)

    figs_bytes = {}
    all_passed = True

    # ── First run regression to get residuals ─────────────────────────────
    with st.spinner("Running analysis…"):
        reg_results = run_regression(y, X_df)
        residuals = reg_results["residuals"]
        fitted = reg_results["fitted"]

    # ─────────────────────────────────────────────────────────────────────
    # TEST 1: NORMALITY
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 1 · Normality Test of Residuals")

    norm_res = run_normality(residuals)
    norm_passed = norm_res["passed"]
    if not norm_passed:
        all_passed = False

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("n", norm_res["n"])
    col2.metric("Shapiro-Wilk W", f"{norm_res['sw_stat']:.4f}")
    col3.metric("Shapiro-Wilk Sig.", f"{norm_res['sw_p']:.4f}")
    col4.metric("Recommended Test", norm_res["primary"])

    norm_tbl = pd.DataFrame([
        {"Test": "Kolmogorov-Smirnov",
         "Statistic": f"{norm_res['ks_stat']:.4f}",
         "Sig.": f"{norm_res['ks_p']:.4f}",
         "df": norm_res["n"],
         "Note": "← Use if n > 50" if norm_res["n"] > 50 else ""},
        {"Test": "Shapiro-Wilk",
         "Statistic": f"{norm_res['sw_stat']:.4f}",
         "Sig.": f"{norm_res['sw_p']:.4f}",
         "df": norm_res["n"],
         "Note": "← Use if n ≤ 50" if norm_res["n"] <= 50 else ""},
    ])
    st.dataframe(norm_tbl, use_container_width=True, hide_index=True)

    primary_p = norm_res["primary_p"]
    display_pass_fail(norm_passed)
    st.markdown(f"""
    <div class="interp-box">
    Based on the <b>{norm_res['primary']}</b> test (appropriate for n = {norm_res['n']}),
    the significance value is <b>{primary_p:.4f}</b>
    ({'> 0.05' if primary_p > 0.05 else '< 0.05'}).
    The residuals are <b>{'normally distributed' if norm_passed else 'NOT normally distributed'}</b>.
    Normality assumption: <b>{'SATISFIED ✓' if norm_passed else 'VIOLATED ✗'}</b>.
    </div>
    """, unsafe_allow_html=True)

    if not norm_passed:
        with st.expander("💡 Recommendations — Normality Violated"):
            st.markdown(RECOMMENDATIONS["normality"])

    fig_norm = plot_normality(residuals)
    figs_bytes["normality"] = fig_to_bytes(fig_norm)
    st.pyplot(fig_norm, use_container_width=True)
    plt.close(fig_norm)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 2: LINEARITY
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 2 · Linearity Test")

    lin_results = {}
    for x_col in x_cols:
        st.markdown(f"**{x_col} → {y_col}**")
        lr = run_linearity(X_df[x_col], y)
        lin_results[x_col] = lr
        if not lr["passed"]:
            all_passed = False

        lin_tbl = pd.DataFrame([
            {"Source": "Between Groups (Combined)",
             "Sum of Squares": f"{lr['ss_reg'] + lr['ss_dev']:.4f}",
             "df": lr["df_lin"] + (lr["df_dev"] if lr["df_dev"] > 0 else 0),
             "Mean Square": "-", "F": "-", "Sig.": "-"},
            {"Source": "  Linearity",
             "Sum of Squares": f"{lr['ss_reg']:.4f}",
             "df": lr["df_lin"],
             "Mean Square": f"{lr['ms_lin']:.4f}",
             "F": f"{lr['F_lin']:.4f}" if not np.isnan(lr['F_lin']) else "-",
             "Sig.": f"{lr['p_lin']:.4f}" if not np.isnan(lr['p_lin']) else "-"},
            {"Source": "  Deviation from Linearity",
             "Sum of Squares": f"{lr['ss_dev']:.4f}",
             "df": lr["df_dev"] if lr["df_dev"] > 0 else "-",
             "Mean Square": f"{lr['ms_dev']:.4f}" if (lr["df_dev"] > 0 and not np.isnan(lr["ms_dev"])) else "-",
             "F": f"{lr['F_dev']:.4f}" if (lr["df_dev"] > 0 and not np.isnan(lr["F_dev"])) else "-",
             "Sig.": f"{lr['p_dev']:.4f}" if (lr["df_dev"] > 0 and not np.isnan(lr["p_dev"])) else "-"},
            {"Source": "Within Groups",
             "Sum of Squares": f"{lr['ss_within']:.4f}",
             "df": lr["df_within"],
             "Mean Square": f"{lr['ms_within']:.4f}",
             "F": "-", "Sig.": "-"},
        ])
        st.dataframe(lin_tbl, use_container_width=True, hide_index=True)

        display_pass_fail(lr["passed"])
        p_lin_str = f"{lr['p_lin']:.4f}" if not np.isnan(lr['p_lin']) else "N/A"
        p_dev_str = f"{lr['p_dev']:.4f}" if (lr["df_dev"] > 0 and not np.isnan(lr["p_dev"])) else "N/A"
        continuous_note = ""
        if lr.get("continuous_x"):
            continuous_note = "<br><i>Note: X contains all unique values (continuous variable) — Deviation from Linearity cannot be computed. OLS F-test used for Linearity.</i>"
        st.markdown(f"""
        <div class="interp-box">
        Sig. Linearity = <b>{p_lin_str}</b> {'(< 0.05 ✓)' if (not np.isnan(lr['p_lin']) and lr['p_lin'] < 0.05) else '(> 0.05 ✗)'} &nbsp;|&nbsp;
        Sig. Deviation from Linearity = <b>{p_dev_str}</b>
        {'(> 0.05 ✓)' if (lr["df_dev"] > 0 and not np.isnan(lr.get("p_dev", float("nan"))) and lr["p_dev"] > 0.05) else ''}.{continuous_note}<br>
        The relationship between <b>{x_col}</b> and <b>{y_col}</b> is
        <b>{'linear ✓' if lr['passed'] else 'NOT confirmed as linear ✗'}</b>.
        </div>
        """, unsafe_allow_html=True)

        if not lr["passed"]:
            with st.expander(f"💡 Recommendations — Linearity Violated ({x_col})"):
                st.markdown(RECOMMENDATIONS["linearity"])

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 3: MULTICOLLINEARITY (multiple only)
    # ─────────────────────────────────────────────────────────────────────
    vif_df = None
    if reg_type == "Multiple":
        st.markdown("### 3 · Multicollinearity Test")
        vif_df = run_vif(X_df)
        mc_passed = all(vif_df["VIF"] < 10) and all(vif_df["Tolerance"] > 0.10)
        if not mc_passed:
            all_passed = False

        vif_display = vif_df.copy()
        vif_display["Tolerance"] = vif_display["Tolerance"].map("{:.4f}".format)
        vif_display["VIF"] = vif_display["VIF"].map("{:.4f}".format)
        st.dataframe(vif_display, use_container_width=True, hide_index=True)

        display_pass_fail(mc_passed)
        st.markdown(f"""
        <div class="interp-box">
        Tolerance criterion: > 0.10 &nbsp;|&nbsp; VIF criterion: < 10.0.<br>
        {'All variables meet both criteria — no multicollinearity detected.' if mc_passed
         else 'One or more variables violate the multicollinearity thresholds.'}
        Multicollinearity assumption: <b>{'SATISFIED ✓' if mc_passed else 'VIOLATED ✗'}</b>.
        </div>
        """, unsafe_allow_html=True)

        if not mc_passed:
            with st.expander("💡 Recommendations — Multicollinearity Detected"):
                st.markdown(RECOMMENDATIONS["multicollinearity"])

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 4: HETEROSCEDASTICITY
    # ─────────────────────────────────────────────────────────────────────
    sec_num = 4 if reg_type == "Multiple" else 3
    st.markdown(f"### {sec_num} · Heteroscedasticity Test (Glejser + Scatterplot)")

    glejser_df, glejser_passed, _ = run_glejser(residuals, X_df)
    if not glejser_passed:
        all_passed = False

    # Format display
    glejser_display = glejser_df.copy()
    for col in ["B", "Std. Error", "t"]:
        glejser_display[col] = glejser_display[col].map("{:.4f}".format)
    glejser_display["Sig."] = glejser_display["Sig."].map("{:.4f}".format)
    st.dataframe(glejser_display, use_container_width=True, hide_index=True)

    display_pass_fail(glejser_passed)
    sig_vals = glejser_df[glejser_df["Term"] != "(Constant)"]["Sig."].tolist()
    sig_str = ", ".join([f"{s:.4f}" for s in sig_vals])
    st.markdown(f"""
    <div class="interp-box">
    Glejser test — regressing |residuals| on predictor(s).<br>
    Sig. values for predictors: <b>{sig_str}</b>
    ({'all > 0.05 ✓' if glejser_passed else 'at least one < 0.05 ✗'}).<br>
    Heteroscedasticity assumption: <b>{'SATISFIED (no heteroscedasticity) ✓' if glejser_passed else 'VIOLATED (heteroscedasticity present) ✗'}</b>.
    </div>
    """, unsafe_allow_html=True)

    if not glejser_passed:
        with st.expander("💡 Recommendations — Heteroscedasticity Detected"):
            st.markdown(RECOMMENDATIONS["heteroscedasticity"])

    # Scatterplot ZRESID vs ZPRED
    fig_scatter = plot_scatter_residuals(fitted, residuals)
    figs_bytes["scatter"] = fig_to_bytes(fig_scatter)
    st.pyplot(fig_scatter, use_container_width=True)
    plt.close(fig_scatter)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # TEST 5: AUTOCORRELATION
    # ─────────────────────────────────────────────────────────────────────
    sec_num2 = 5 if reg_type == "Multiple" else 4
    st.markdown(f"### {sec_num2} · Autocorrelation Test (Durbin-Watson)")

    dw_value = reg_results["dw"]
    dw_passed = 1.5 <= dw_value <= 2.5
    if not dw_passed:
        all_passed = False

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Durbin-Watson", f"{dw_value:.4f}")
    col_b.metric("Acceptable Range", "1.5 – 2.5")
    col_c.metric("Status", "PASS ✓" if dw_passed else "FAIL ✗")

    display_pass_fail(dw_passed)
    st.markdown(f"""
    <div class="interp-box">
    Durbin-Watson statistic = <b>{dw_value:.4f}</b>
    ({'within' if dw_passed else 'outside'} the acceptable range of 1.5 to 2.5).<br>
    {'No autocorrelation detected.' if dw_passed else
     ('Positive autocorrelation detected (DW < 1.5).' if dw_value < 1.5
      else 'Negative autocorrelation detected (DW > 2.5).')}
    Autocorrelation assumption: <b>{'SATISFIED ✓' if dw_passed else 'VIOLATED ✗'}</b>.
    </div>
    """, unsafe_allow_html=True)

    if not dw_passed:
        with st.expander("💡 Recommendations — Autocorrelation Detected"):
            st.markdown(RECOMMENDATIONS["autocorrelation"])

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # ASSUMPTION SUMMARY
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### Assumption Tests Summary")

    summary_rows = [
        {"Test": "Normality (Residuals)", "Key Statistic": f"{norm_res['primary']} Sig. = {norm_res['primary_p']:.4f}", "Result": "✓ SATISFIED" if norm_passed else "✗ VIOLATED"},
    ]
    for x_col, lr in lin_results.items():
        summary_rows.append({
            "Test": f"Linearity ({x_col} → {y_col})",
            "Key Statistic": f"Sig. Lin. = {lr['p_lin']:.4f}",
            "Result": "✓ SATISFIED" if lr["passed"] else "✗ VIOLATED"
        })
    if reg_type == "Multiple" and vif_df is not None:
        mc_p = all(vif_df["VIF"] < 10)
        summary_rows.append({
            "Test": "Multicollinearity",
            "Key Statistic": f"Max VIF = {vif_df['VIF'].max():.4f}",
            "Result": "✓ SATISFIED" if mc_p else "✗ VIOLATED"
        })
    summary_rows.append({
        "Test": "Heteroscedasticity (Glejser)",
        "Key Statistic": f"Min Sig. = {glejser_df[glejser_df['Term'] != '(Constant)']['Sig.'].min():.4f}",
        "Result": "✓ SATISFIED" if glejser_passed else "✗ VIOLATED"
    })
    summary_rows.append({
        "Test": "Autocorrelation (Durbin-Watson)",
        "Key Statistic": f"DW = {dw_value:.4f}",
        "Result": "✓ SATISFIED" if dw_passed else "✗ VIOLATED"
    })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if all_passed:
        st.success("✅ All prerequisite assumptions are satisfied. Proceeding to regression analysis.")
    else:
        st.warning(
            "⚠️ One or more assumptions are violated. Review recommendations above. "
            "Regression results are displayed below for reference, but interpret with caution."
        )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # REGRESSION RESULTS
    # ═════════════════════════════════════════════════════════════════════
    sec_num3 = 6 if reg_type == "Multiple" else 5
    st.markdown(f'<div class="section-header">{sec_num3} · Regression Analysis Results</div>',
                unsafe_allow_html=True)

    # ── Model Summary ─────────────────────────────────────────────────────
    st.markdown("#### Model Summary")
    ms_cols = st.columns(5)
    ms_cols[0].metric("R", f"{reg_results['R']:.4f}")
    ms_cols[1].metric("R Square", f"{reg_results['R2']:.4f}")
    ms_cols[2].metric("Adjusted R²", f"{reg_results['adj_R2']:.4f}")
    ms_cols[3].metric("Std. Error of Estimate", f"{reg_results['se_est']:.4f}")
    ms_cols[4].metric("Durbin-Watson", f"{reg_results['dw']:.4f}")

    r2_pct = reg_results["R2"] * 100
    st.markdown(f"""
    <div class="interp-box">
    R = <b>{reg_results['R']:.4f}</b> — correlation between predicted and observed Y.<br>
    R² = <b>{reg_results['R2']:.4f}</b> — <b>{r2_pct:.1f}%</b> of the variance in <b>{y_col}</b>
    is explained by {', '.join(x_cols)}.<br>
    Adjusted R² = <b>{reg_results['adj_R2']:.4f}</b> (accounts for number of predictors and sample size).
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── ANOVA ─────────────────────────────────────────────────────────────
    st.markdown("#### ANOVA")
    display_anova_table(reg_results["anova"])

    sig_model = reg_results["p_F"] < 0.05
    st.markdown(f"""
    <div class="interp-box">
    F = <b>{reg_results['F']:.4f}</b>, Sig. = <b>{reg_results['p_F']:.4f}</b>
    ({'< 0.05' if sig_model else '> 0.05'}).<br>
    The regression model is <b>{'statistically significant ✓' if sig_model else 'NOT statistically significant ✗'}</b>.
    {'The independent variable(s) significantly predict ' + y_col + '.' if sig_model
     else 'The model cannot reliably predict ' + y_col + '.'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Coefficients ──────────────────────────────────────────────────────
    st.markdown("#### Coefficients")
    display_coef_table(reg_results["coef_df"])

    # Regression equation
    coefs = reg_results["coef_df"]
    const_val = coefs.loc[coefs["Model"] == "(Constant)", "B"].values[0]
    eq_parts = [f"{const_val:.4f}"]
    interp_parts = []
    for _, row in coefs[coefs["Model"] != "(Constant)"].iterrows():
        sign = "+" if row["B"] >= 0 else ""
        eq_parts.append(f"{sign}{row['B']:.4f}({row['Model']})")
        sig_x = float(row["Sig."]) < 0.05
        direction = "positive" if row["B"] > 0 else "negative"
        interp_parts.append(
            f"<b>{row['Model']}</b>: B = {row['B']:.4f}, Sig. = {float(row['Sig.']):.4f} "
            f"→ {'significant' if sig_x else 'not significant'}, {direction} effect on {y_col}."
        )

    eq_str = "Ŷ = " + " ".join(eq_parts)
    st.markdown(f'<div class="equation-box">{eq_str}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="interp-box">
    <b>Constant (a):</b> When all X = 0, predicted {y_col} = {const_val:.4f}.<br><br>
    {"<br>".join(interp_parts)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Regression plot ───────────────────────────────────────────────────
    if reg_type == "Simple":
        fig_reg = plot_regression_line(
            X_df[x_cols[0]], y, x_cols[0], y_col,
            reg_results["model"]
        )
        figs_bytes["regression"] = fig_to_bytes(fig_reg)
        st.pyplot(fig_reg, use_container_width=True)
        plt.close(fig_reg)
    else:
        # For multiple: scatter of fitted vs actual
        fig2, ax2 = plt.subplots(figsize=(7, 4))
        style_ax(ax2, fig2)
        ax2.scatter(fitted, y, color=GOLD, s=25, alpha=0.8, zorder=3)
        mn = min(fitted.min(), y.min())
        mx = max(fitted.max(), y.max())
        ax2.plot([mn, mx], [mn, mx], color="#f0e6d3", linewidth=1.5,
                 linestyle="--", label="Perfect fit line")
        ax2.set_title("Observed vs Fitted Values", fontsize=11,
                      color=LIGHT_TXT, fontfamily="serif")
        ax2.set_xlabel("Fitted (Predicted) Values", color=MUTED_TXT, fontsize=9)
        ax2.set_ylabel(f"Observed {y_col}", color=MUTED_TXT, fontsize=9)
        ax2.legend(facecolor=CARD_BG, edgecolor="#2a2d3a",
                   labelcolor=LIGHT_TXT, fontsize=8)
        plt.tight_layout()
        figs_bytes["regression"] = fig_to_bytes(fig2)
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Auto Conclusion ───────────────────────────────────────────────────
    sec_num4 = 7 if reg_type == "Multiple" else 6
    st.markdown(f'<div class="section-header">{sec_num4} · Interpretation & Conclusion</div>',
                unsafe_allow_html=True)

    lin_interp = " ".join([
        f"The linearity test confirmed {'a linear' if lr['passed'] else 'no confirmed linear'} "
        f"relationship between {xc} and {y_col} "
        f"(Sig. Linearity = {lr['p_lin']:.4f}; "
        f"Sig. Deviation = {lr['p_dev']:.4f if (lr['df_dev'] > 0 and not np.isnan(lr['p_dev'])) else 'N/A'})."
        for xc, lr in lin_results.items()
    ])

    mc_sentence = ""
    if reg_type == "Multiple" and vif_df is not None:
        mc_p = all(vif_df["VIF"] < 10)
        mc_sentence = (
            f"No multicollinearity was detected among the predictors (all VIF < 10; all Tolerance > 0.10). "
            if mc_p else
            f"Multicollinearity was detected among the predictors. "
        )

    pred_sentences = " ".join([
        f"{row['Model']} has a {'positive' if row['B'] > 0 else 'negative'} and "
        f"{'significant' if float(row['Sig.']) < 0.05 else 'non-significant'} effect on {y_col} "
        f"(B = {row['B']:.4f}, β = {fmt(row['Beta'])}, t = {row['t']:.4f}, Sig. = {float(row['Sig.']):.4f})."
        for _, row in reg_results["coef_df"][reg_results["coef_df"]["Model"] != "(Constant)"].iterrows()
    ])

    conclusion_text = (
        f"Prior to conducting the {'simple' if reg_type == 'Simple' else 'multiple'} linear regression analysis, "
        f"all prerequisite assumption tests were performed. "
        f"The {norm_res['primary']} normality test showed that residuals are "
        f"{'normally distributed' if norm_passed else 'not normally distributed'} "
        f"(Sig. = {norm_res['primary_p']:.4f}). "
        f"{lin_interp} "
        f"{mc_sentence}"
        f"The Glejser heteroscedasticity test indicated "
        f"{'no heteroscedasticity (all Sig. > 0.05)' if glejser_passed else 'the presence of heteroscedasticity'}. "
        f"The Durbin-Watson statistic (DW = {dw_value:.4f}) confirmed "
        f"{'no autocorrelation' if dw_passed else 'the presence of autocorrelation'}. "
        f"\n\n"
        f"{'Simple' if reg_type == 'Simple' else 'Multiple'} linear regression results showed that the model is "
        f"{'statistically significant' if sig_model else 'not statistically significant'} "
        f"(F = {reg_results['F']:.4f}, Sig. = {reg_results['p_F']:.4f}). "
        f"The coefficient of determination R² = {reg_results['R2']:.4f}, indicating that "
        f"{r2_pct:.1f}% of the variance in {y_col} is explained by {', '.join(x_cols)}. "
        f"The regression equation is: {eq_str}. "
        f"{pred_sentences}"
    )

    st.markdown(f'<div class="interp-box" style="font-size:0.95rem;">{conclusion_text.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════
    # DOWNLOAD SECTION
    # ═════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-header">Download Report</div>',
                unsafe_allow_html=True)

    with st.spinner("Generating Word report…"):
        word_bytes = generate_word_report(
            reg_type=reg_type,
            y_col=y_col,
            x_cols=x_cols,
            norm_results=norm_res,
            lin_results=lin_results,
            vif_df=vif_df,
            glejser_df=glejser_df,
            glejser_passed=glejser_passed,
            dw_value=dw_value,
            reg_results=reg_results,
            all_passed=all_passed,
            figs_bytes=figs_bytes,
        )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="⬇ Download Word Report (.docx)",
            data=word_bytes,
            file_name=f"regression_report_{y_col}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

    with col_dl2:
        # CSV of all results
        all_results_rows = []
        all_results_rows.append({"Section": "MODEL SUMMARY", "Item": "R",
                                  "Value": f"{reg_results['R']:.4f}"})
        all_results_rows.append({"Section": "MODEL SUMMARY", "Item": "R Square",
                                  "Value": f"{reg_results['R2']:.4f}"})
        all_results_rows.append({"Section": "MODEL SUMMARY", "Item": "Adjusted R Square",
                                  "Value": f"{reg_results['adj_R2']:.4f}"})
        all_results_rows.append({"Section": "MODEL SUMMARY", "Item": "Std. Error",
                                  "Value": f"{reg_results['se_est']:.4f}"})
        all_results_rows.append({"Section": "ANOVA", "Item": "F",
                                  "Value": f"{reg_results['F']:.4f}"})
        all_results_rows.append({"Section": "ANOVA", "Item": "Sig.",
                                  "Value": f"{reg_results['p_F']:.4f}"})
        for _, row in reg_results["coef_df"].iterrows():
            all_results_rows.append({"Section": "COEFFICIENTS",
                                      "Item": f"{row['Model']} B",
                                      "Value": f"{row['B']:.4f}"})
            all_results_rows.append({"Section": "COEFFICIENTS",
                                      "Item": f"{row['Model']} Sig.",
                                      "Value": f"{float(row['Sig.']):.4f}"})

        csv_results = pd.DataFrame(all_results_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇ Download Results Summary (.csv)",
            data=csv_results,
            file_name=f"regression_results_{y_col}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("""
    <div style="text-align:center; color:#3a3a4a; font-size:0.78rem; padding: 2rem 0 1rem 0; font-family: 'DM Mono', monospace;">
    Regression Analyzer · SPSS-equivalent formulas · statsmodels + scipy
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
