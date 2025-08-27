# daily_exec_brief.py
# Final version with improved CSS + labeled chart axes

import os
import json
import math
import urllib.parse
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------
# CONFIG
# ------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME   = "llama-3.3-70b-versatile"

CURRENCY      = "AED"
CURRENCY_SYM  = "AED "
DECIMALS      = 2

ORDERS_CSV    = "orders.csv"

DATE_COL_CANDIDATES   = ["order_date", "created_at", "date"]
TOTAL_COL_CANDIDATES  = ["total", "amount", "order_total", "grand_total"]
ORDER_ID_CANDIDATES   = ["order_id", "id", "order_number"]

# Chart size (bigger for clarity)
CHART_WIDTH   = 900
CHART_HEIGHT  = 300

# Email config (from .env)
FROM_EMAIL   = os.getenv("GMAIL_ADDRESS")
EMAIL_PASS   = os.getenv("GMAIL_APP_PASSWORD")
TO_EMAIL     = os.getenv("GMAIL_ADDRESS")  # send to self for now

# ------------------------------
# HELPERS
# ------------------------------
def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
        # case-insensitive match
        lc = [col.lower() for col in df.columns]
        if c.lower() in lc:
            return df.columns[lc.index(c.lower())]
    raise KeyError(f"None of {candidates} found. Available: {list(df.columns)}")

def _to_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def load_orders(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    date_col  = _pick_col(df, DATE_COL_CANDIDATES)
    total_col = _pick_col(df, TOTAL_COL_CANDIDATES)
    df = df.copy()
    df[date_col]  = _to_date_series(df[date_col])
    df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
    df = df.dropna(subset=[date_col, total_col])
    return df.rename(columns={date_col: "date", total_col: "total"})

def kpis(df: pd.DataFrame) -> dict:
    orders = len(df)
    revenue = float(df["total"].sum()) if orders else 0.0
    aov = (revenue / orders) if orders else 0.0
    return {"orders": orders, "revenue": revenue, "aov": aov}

def pct_change(cur: float, base: float) -> float:
    if base == 0:
        return math.inf if cur > 0 else 0.0
    return (cur - base) / base * 100.0

def mad_based_zscores(values):
    v = pd.Series(values).astype(float)
    med = v.median()
    mad = (v - med).abs().median()
    if mad == 0:
        return pd.Series([0.0] * len(v))
    return 0.6745 * (v - med) / mad

def detect_anomalies(y_rev, y_aov, baseline_daily):
    # Safe guard: no baseline
    if baseline_daily.empty:
        return False, "No baseline data"

    rev_series = baseline_daily["revenue"]
    aov_series = baseline_daily["aov"]

    # Append yesterday for the z-score position
    rev_series = pd.concat([rev_series, pd.Series([y_rev])], ignore_index=True)
    aov_series = pd.concat([aov_series, pd.Series([y_aov])], ignore_index=True)

    rev_z = float(mad_based_zscores(rev_series).iloc[-1])
    aov_z = float(mad_based_zscores(aov_series).iloc[-1])

    anomaly = (abs(rev_z) >= 3) or (abs(aov_z) >= 3)
    reasons = []
    if abs(rev_z) >= 3:
        reasons.append(f"Revenue z≈{rev_z:.1f}")
    if abs(aov_z) >= 3:
        reasons.append(f"AOV z≈{aov_z:.1f}")
    return anomaly, ", ".join(reasons) if reasons else "—"

def agg_daily(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("date").agg(revenue=("total", "sum"),
                               orders=("total", "count"))
    g["aov"] = g["revenue"] / g["orders"].replace(0, pd.NA)
    return g.reset_index().sort_values("date")

def chart_url_with_axes(series_dates, series_values):
    """
    Chart.js via QuickChart:
    - Baseline area (all points except yesterday) shaded light blue
    - Main revenue line across entire period
    - Yesterday marker (red dot)
    - Last segment area (prev→yesterday) filled GREEN if rising, RED if falling
    """
    labels = [d.strftime("%b %d") for d in series_dates]
    data = [round(float(x), 2) for x in series_values]

    # Baseline = everything before the last point (yesterday)
    baseline_len = max(len(data) - 1, 0)
    baseline_data = [data[i] if i < baseline_len else None for i in range(len(data))]

    # Yesterday marker (only last point)
    yesterday_only = [None] * baseline_len + ([data[-1]] if len(data) else [])

    # Rising/declining last segment color
    if len(data) >= 2:
        prev_y, last_y = data[-2], data[-1]
        rising = last_y >= prev_y
        seg_bg = "rgba(22,163,74,0.22)" if rising else "rgba(220,38,38,0.22)"  # green/red
        # Dataset that defines ONLY the last segment; Chart.js will fill under it
        seg_data = [None] * (baseline_len - 1) + [prev_y, last_y] if baseline_len >= 1 else []
    else:
        seg_bg = "rgba(0,0,0,0)"  # no segment
        seg_data = []

    config = {
        "type": "line",
        "data": {
            "labels": labels,
            "datasets": [
                # A) Shaded baseline area
                {
                    "label": "Baseline",
                    "data": baseline_data,
                    "fill": "origin",
                    "borderWidth": 1,
                    "borderColor": "#93c5fd",
                    "backgroundColor": "rgba(59,130,246,0.18)",
                    "tension": 0.3,
                    "pointRadius": 0,
                    "order": 3
                },
                # B) Last segment area (green/red)
                {
                    "label": "Last Segment",
                    "data": seg_data,
                    "fill": "origin",
                    "borderWidth": 0,
                    "backgroundColor": seg_bg,
                    "tension": 0.3,
                    "pointRadius": 0,
                    "spanGaps": True,
                    "order": 2
                },
                # C) Main line (across entire series)
                {
                    "label": "Revenue",
                    "data": data,
                    "fill": False,
                    "borderWidth": 2,
                    "borderColor": "#3b82f6",
                    "tension": 0.3,
                    "pointRadius": 0,
                    "order": 1
                },
                # D) Yesterday marker
                {
                    "label": "Yesterday",
                    "data": yesterday_only,
                    "showLine": False,
                    "pointRadius": 6,
                    "pointHoverRadius": 7,
                    "pointBackgroundColor": "#ef4444",
                    "pointBorderColor": "#ef4444",
                    "order": 0
                }
            ]
        },
        "options": {
            "plugins": {"legend": False},
            "layout": {"padding": 8},
            "scales": {
                "x": {
                    "title": {"display": True, "text": "Date"},
                    "ticks": {"maxTicksLimit": 12, "autoSkip": True}
                },
                "y": {
                    "title": {"display": True, "text": f"Revenue ({CURRENCY})"},
                    "beginAtZero": True,
                    "ticks": {"precision": 0}
                }
            }
        }
    }

    c = urllib.parse.quote(json.dumps(config, separators=(",", ":")))
    return f"https://quickchart.io/chart?c={c}&width={CHART_WIDTH}&height={CHART_HEIGHT}&backgroundColor=transparent"



def format_money(x: float) -> str:
    return f"{CURRENCY_SYM}{x:,.{DECIMALS}f}"

# ------------------------------
# LLM
# ------------------------------
def build_llm():
    if not GROQ_API_KEY:
        raise EnvironmentError("Missing GROQ_API_KEY in .env")
    return ChatGroq(model=MODEL_NAME, temperature=0.2, api_key=GROQ_API_KEY)

# A more sophisticated prompt that encourages interpretation

BRIEF_TMPL = PromptTemplate.from_template(
    """You are an executive reporting assistant tasked with writing a detailed business summary.
Your response MUST be a single, well-articulated paragraph of 100 - 120 words.

**Instructions:**
1.  Start by reporting yesterday's key performance metrics: revenue and AOV.
2.  Compare these figures to the baseline, explicitly mentioning the percentage changes.
3.  **Crucially, interpret the relationship between the revenue and AOV changes.** For example, what does it mean if revenue is down but AOV is up?
4.  If an anomaly was detected, state the reason.
5.  Conclude with one practical, business-friendly action item that directly addresses the key finding from your interpretation.

Company currency: {currency}.

**Data for your summary:**
- Yesterday: revenue={y_rev}, aov={y_aov}, orders={y_orders}
- Baseline (30-day avg): revenue={b_rev_mean}, aov={b_aov_mean}, orders/day={b_orders_mean}
- Performance Delta: revenue_pct={rev_pct:.1f}%, aov_pct={aov_pct:.1f}%
- Anomaly Detected: {anomaly_flag} (Reason: {anomaly_reason})
"""
)

def write_brief(llm, context: dict) -> str:
    chain = BRIEF_TMPL | llm | StrOutputParser()
    return chain.invoke(context).strip()

# ------------------------------
# HTML email assembly (with nicer CSS)
# ------------------------------
def html_email(kpi_y, kpi_b_means, deltas, chart_url, brief_text, anomaly_flag, anomaly_reason):
    # Colors
    green = "#16a34a"
    red   = "#dc2626"
    gray  = "#6b7280"
    text  = "#0f172a"

    # Badge color based on delta sign
    rev_badge_color = green if deltas['rev_pct'] >= 0 else red
    aov_badge_color = green if deltas['aov_pct'] >= 0 else red

    return f"""
<!DOCTYPE html>
<html>
  <body style="margin:0;padding:24px;background:#f6f7fb;">
    <div style="
      max-width:820px;margin:0 auto;background:#ffffff;
      border-radius:16px;box-shadow:0 10px 25px rgba(2,6,23,.08);
      overflow:hidden;font-family:Inter,Segoe UI,Arial,sans-serif;color:{text};
    ">
      <div style="background:linear-gradient(135deg,#eef2ff,#f0f9ff);padding:20px 24px;border-bottom:1px solid #e5e7eb;">
        <h1 style="margin:0;font-size:20px;letter-spacing:.2px;">Daily Revenue & Anomaly Brief</h1>
        <div style="margin-top:6px;font-size:13px;color:{gray};">Currency: {CURRENCY}</div>
      </div>

      <div style="padding:24px;">
        <!-- KPI row -->
        <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-bottom:16px;">
          <div style="display:inline-flex;align-items:baseline;gap:8px;padding:12px 16px;border-radius:12px;background:#f8fafc;border:1px solid #eef2f7;">
            <div style="font-size:22px;font-weight:700;">{format_money(kpi_y['revenue'])}</div>
            <div style="font-size:13px;color:{gray};">Revenue</div>
          </div>
          <div style="display:inline-flex;align-items:baseline;gap:8px;padding:12px 16px;border-radius:12px;background:#f8fafc;border:1px solid #eef2f7;">
            <div style="font-size:22px;font-weight:700;">{format_money(kpi_y['aov'])}</div>
            <div style="font-size:13px;color:{gray};">AOV</div>
          </div>
          <div style="display:inline-flex;align-items:baseline;gap:8px;padding:12px 16px;border-radius:12px;background:#f8fafc;border:1px solid #eef2f7;">
            <div style="font-size:20px;font-weight:700;">{kpi_y['orders']}</div>
            <div style="font-size:13px;color:{gray};">Orders</div>
          </div>
        </div>

        <!-- Deltas -->
        <div style="margin-bottom:18px;font-size:14px;color:{text};">
          <span style="margin-right:10px;font-weight:600;">vs Baseline</span>
          <span style="display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(22,163,74,.08);border:1px solid {rev_badge_color};color:{rev_badge_color};font-weight:600;margin-right:6px;">
            Revenue: {deltas['rev_pct']:+.1f}%
          </span>
          <span style="display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(22,163,74,.08);border:1px solid {aov_badge_color};color:{aov_badge_color};font-weight:600;">
            AOV: {deltas['aov_pct']:+.1f}%
          </span>
        </div>

        <!-- Chart -->
        <img alt="Revenue over time" src="{chart_url}" width="{CHART_WIDTH}" height="{CHART_HEIGHT}" style="width:100%;height:auto;border:1px solid #eef2f7;border-radius:12px;display:block;margin:8px 0 18px 0;" />

        <!-- Brief -->
        <div style="font-size:15px;line-height:1.55;color:{text};margin-bottom:8px;">
          {brief_text}
        </div>

        <!-- Anomaly -->
        <div style="margin-top:8px;font-size:14px;font-weight:700;color:{green if not anomaly_flag else red};">
          {'No anomaly detected' if not anomaly_flag else 'Anomaly detected'}{'' if (not anomaly_flag or not anomaly_reason or anomaly_reason=='—') else f' — {anomaly_reason}'}
        </div>
      </div>

      <div style="padding:14px 24px;border-top:1px solid #e5e7eb;background:#fafafa;color:{gray};font-size:12px;">
        Automated report • {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}
      </div>
    </div>
  </body>
</html>
""".strip()

# ------------------------------
# EMAIL SENDER
# ------------------------------
def send_email(html_body, subject="Daily Exec Brief"):
    if not FROM_EMAIL or not EMAIL_PASS or not TO_EMAIL:
        print("⚠️ Missing GMAIL_ADDRESS or GMAIL_APP_PASSWORD in .env — skipping send")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(FROM_EMAIL, EMAIL_PASS)
        server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())

    print(f"✅ Email sent to {TO_EMAIL}")

# ------------------------------
# MAIN
# ------------------------------
def main():
    df = load_orders(ORDERS_CSV)

    yesterday = datetime.today().date() - timedelta(days=1)
    baseline_start = yesterday - timedelta(days=30)
    baseline_end   = yesterday - timedelta(days=1)

    df_y = df[df["date"] == yesterday]
    df_b = df[(df["date"] >= baseline_start) & (df["date"] <= baseline_end)]

    daily_b = agg_daily(df_b)

    kpi_y = kpis(df_y)
    b_means = {
        "revenue": float(daily_b["revenue"].mean()) if len(daily_b) else 0.0,
        "aov": float(daily_b["aov"].mean()) if len(daily_b) else 0.0,
        "orders": float(daily_b["orders"].mean()) if len(daily_b) else 0.0,
    }

    deltas = {
        "rev_pct": pct_change(kpi_y["revenue"], b_means["revenue"]),
        "aov_pct": pct_change(kpi_y["aov"], b_means["aov"]),
    }

    anomaly_flag, anomaly_reason = detect_anomalies(
        kpi_y["revenue"], kpi_y["aov"], daily_b
    )

    # Build labeled chart (dates + values; include yesterday at the end)
    series_dates  = daily_b["date"].tolist()
    series_values = daily_b["revenue"].tolist()
    # append yesterday for visibility even if not in baseline:
    series_dates.append(yesterday)
    series_values.append(kpi_y["revenue"])
    chart_url = chart_url_with_axes(series_dates, series_values)

    llm = build_llm()
    brief = write_brief(llm, {
        "currency": CURRENCY,
        "y_rev": format_money(kpi_y["revenue"]),
        "y_aov": format_money(kpi_y["aov"]),
        "y_orders": kpi_y["orders"],
        "b_rev_mean": format_money(b_means["revenue"]),
        "b_aov_mean": format_money(b_means["aov"]),
        "b_orders_mean": f"{b_means['orders']:.1f}",
        "rev_pct": deltas["rev_pct"],
        "aov_pct": deltas["aov_pct"],
        "anomaly_flag": "Yes" if anomaly_flag else "No",
        "anomaly_reason": anomaly_reason or "—",
    })

    html = html_email(kpi_y, b_means, deltas, chart_url, brief, anomaly_flag, anomaly_reason)

    with open("daily_exec_brief.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("\n✅ Saved preview: daily_exec_brief.html")

    send_email(html, subject="Daily Revenue & Anomaly Brief")

if __name__ == "__main__":
    main()
