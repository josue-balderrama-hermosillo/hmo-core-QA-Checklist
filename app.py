import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt  # charts

# -------------------------
# Fixed assets & config
# -------------------------
LOGO_LEFT = "https://hermosillo.com/wp-content/uploads/2019/08/horizontal-hermosillo-experience-matters-registered-logo.png"
EXCEL_PATH = Path("sample_data") / "QA Check List.xlsx"

st.set_page_config(page_title="Hermosillo • QA Checklist", page_icon="🟧", layout="wide")

# Brand palette
BRAND_ORANGE = "#FF6A00"
INK = "#0F172A"
INK_SOFT = "#334155"
PAPER = "#FFFFFF"
PAPER_ALT = "#F6F7FA"
STROKE = "#E5E7EB"

# -------------------------
# Global CSS (safe top area + full-bleed bands + wider sidebar)
# -------------------------
CSS = f"""
<style>
:root {{
  --ink: {INK};
  --ink-soft: {INK_SOFT};
  --paper: {PAPER};
  --paper-alt: {PAPER_ALT};
  --stroke: {STROKE};
  --brand: {BRAND_ORANGE};
}}

html, body, .stApp {{ background: var(--paper); color: var(--ink); }}

/* Center column spacing and width */
.block-container {{
  padding-top: 4.2rem;   /* space so any top controls (e.g., Deploy) won't overlap */
  max-width: 1200px;
}}

/* Make the sidebar a bit wider and visually distinct */
section[data-testid="stSidebar"] > div:first-child {{
  width: 360px;
  min-width: 360px;
  background: var(--paper-alt);
  border-right: 1px solid var(--stroke);
}}
@media (max-width: 1200px) {{
  section[data-testid="stSidebar"] > div:first-child {{ width: 320px; min-width: 320px; }}
}}

.sidebar-card {{
  background: var(--paper);
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 12px 14px;
  margin-bottom: 12px;
}}
.sidebar-title {{
  font-weight: 800; margin-bottom: .35rem;
}}
.sidebar-sub {{
  color: var(--ink-soft); font-size: .9rem; margin-bottom: .5rem;
}}
.smallnote {{
  color: var(--ink-soft); font-size: .8rem;
}}

.band {{
  position: relative;
  background: var(--paper-alt);
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 16px 18px;
  width: calc(100% + 5rem);
  left: -2.5rem;
  overflow: hidden;
  box-sizing: border-box;
  margin: 10px 0;
}}

.topbar {{
  background: #FFFFFF;
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 12px 16px;
  box-shadow: 0 1px 0 rgba(15,23,42,.04);
  display: flex;
  align-items: center;
  gap: 16px;
  margin: 8px 0 6px 0;
}}
.top-title {{ font-weight: 800; letter-spacing: .2px; color: var(--ink); line-height: 1.1; font-size: 1.15rem; }}
.top-sub {{ color: var(--ink-soft); font-weight: 600; font-size: .9rem; }}

.card {{
  background: var(--paper);
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 16px 18px;
}}
.kpi-label {{ font-size: .72rem; letter-spacing: .08em; color: var(--ink-soft); text-transform: uppercase; }}
.kpi-value {{ font-size: 2rem; font-weight: 800; color: var(--ink); }}

.section-header {{
  margin-top: .6rem;
  padding: .55rem .8rem;
  background: color-mix(in srgb, var(--brand) 10%, transparent);
  border: 1px dashed color-mix(in srgb, var(--brand) 35%, transparent);
  border-left: 4px solid var(--brand);
  border-radius: 10px;
  font-weight: 700;
}}
.row {{ padding:.5rem .3rem; border-bottom:1px dashed var(--stroke); color: var(--ink); }}

.badge {{
  display:inline-block; padding:.2rem .55rem; border-radius:999px;
  background: color-mix(in srgb, var(--brand) 15%, transparent);
  border:1px solid color-mix(in srgb, var(--brand) 35%, transparent);
  color: var(--ink); font-size: .8rem;
}}

.stButton>button, .stDownloadButton>button {{
  border-radius: 10px !important;
  border: 1px solid var(--stroke) !important;
}}
.stDownloadButton>button {{ background: var(--brand) !important; color: #fff !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -------------------------
# TOP BANNER
# -------------------------
st.markdown(
    f"""
    <div class="topbar">
      <img src="{LOGO_LEFT}" style="height:38px"/>
      <div>
        <div class="top-title">Core Innovation • QA Checklist</div>
        <div class="top-sub">Hermosillo — Experience Matters</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Data loader
# -------------------------
def load_excel(file_or_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_or_path, sheet_name="Checklist")
    df.columns = [str(c).strip() for c in df.columns]

    # Column roles: A=code (often "Unnamed: 0"), B=description, C/D=roles
    code_col = next((c for c in df.columns if "unnamed" in c.lower()), df.columns[0])
    role_cols = ["MODELADOR", "COORDINADOR"]
    candidates = [c for c in df.columns if c not in role_cols and c != code_col]
    desc_col = candidates[0] if candidates else df.columns[1]

    # Robust 0/1 normalization
    def _to01(series: pd.Series) -> pd.Series:
        s = series.copy().astype(str).str.strip().str.lower()
        s = s.replace({
            "true": 1, "false": 0, "yes": 1, "no": 0, "y": 1, "n": 0,
            "checked": 1, "unchecked": 0, "nan": np.nan, "none": np.nan, "": np.nan
        })
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        return (s > 0.5).astype(int)

    for rc in role_cols:
        if rc in df.columns: df[rc] = _to01(df[rc])
        else: df[rc] = 0

    # Sections & headers
    df[code_col] = df[code_col].astype(str).replace({"nan": np.nan})
    df["section_code"] = df[code_col].fillna(method="ffill").astype(str).str.strip()
    df["is_header_row"] = df[code_col].notna()

    df["description"] = df[desc_col].astype(str)
    titles = df.loc[df["is_header_row"], ["section_code", "description"]].dropna()
    section_map = dict(zip(titles["section_code"], titles["description"].astype(str)))

    # Discipline by code prefix
    def infer_discipline(code: str) -> str:
        if isinstance(code, str) and re.match(r"^[Cc]\d+", code): return "Civil"
        if isinstance(code, str) and re.match(r"^[Aa]\d+", code): return "Arquitectural"
        return "Unassigned"

    # Skip banners (all caps lines without code)
    def looks_like_banner(text: str) -> bool:
        t = re.sub(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ ]", "", str(text)).strip()
        return bool(t) and t.upper() == t and 3 <= len(t) <= 40

    df["Discipline"] = df["section_code"].apply(infer_discipline)
    df["is_item"] = df["description"].str.strip().ne("") & (~df["is_header_row"]) & (~df["description"].apply(looks_like_banner))

    items = df[df["is_item"]].copy()
    items["section_title"] = items["section_code"].map(section_map).fillna(items["section_code"])
    items["item_id"] = items.index.astype(str)
    items = items[["item_id","section_code","section_title","Discipline","description","MODELADOR","COORDINADOR"]]
    return items

if not EXCEL_PATH.exists():
    st.error(f"Bundled file not found: {EXCEL_PATH}")
    st.stop()

items = load_excel(EXCEL_PATH)

# Keep check state between reruns
if "checks" not in st.session_state:
    st.session_state["checks"] = {
        r["item_id"]: {"MODELADOR": int(r["MODELADOR"]), "COORDINADOR": int(r["COORDINADOR"])}
        for _, r in items.iterrows()
    }

# -------------------------
# CONTROLS (main area)
# -------------------------
st.markdown('<div class="band">', unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    st.write("**Discipline**")
    discipline = st.segmented_control(
        "Discipline", options=["All","Civil","Arquitectural"],
        selection_mode="single", default="All", label_visibility="collapsed",
    )
with c2:
    st.write("**Search**")
    search = st.text_input("", placeholder="Search items…", label_visibility="collapsed")
with c3:
    st.write("**Actions**")
    st.caption("Use section bulk toggles below")
st.markdown('</div>', unsafe_allow_html=True)

# Apply filters
view = items.copy()
if discipline != "All":
    view = view[view["Discipline"] == discipline]
if search:
    view = view[view["description"].str.contains(search, case=False, na=False)]

# -------------------------
# Metrics helpers
# -------------------------
def compute_metrics(df: pd.DataFrame, checks: dict):
    total = len(df)
    if total == 0: return 0,0,0,0,0.0,0.0,0.0
    m_checked = sum(checks[r["item_id"]]["MODELADOR"] for _, r in df.iterrows())
    c_checked = sum(checks[r["item_id"]]["COORDINADOR"] for _, r in df.iterrows())
    both_checked = sum(
        1 if (checks[r["item_id"]]["MODELADOR"]==1 and checks[r["item_id"]]["COORDINADOR"]==1) else 0
        for _, r in df.iterrows()
    )
    return total, m_checked, c_checked, both_checked, m_checked/total, c_checked/total, both_checked/total

tot, m_chk, c_chk, both_chk, m_pct, c_pct, both_pct = compute_metrics(view, st.session_state["checks"])

# -------------------------
# KPIs (main area)
# -------------------------
st.markdown('<div class="band">', unsafe_allow_html=True)
k1, k2, k3 = st.columns([1,1,1])
with k1:
    st.markdown(f'<div class="card"><div class="kpi-label">Items (View)</div><div class="kpi-value">{tot}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="card"><div class="kpi-label">Modelador ✓</div><div class="kpi-value">{m_chk}/{tot}</div><div class="kpi-label">{m_pct*100:.1f}%</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="card"><div class="kpi-label">Coordinador ✓</div><div class="kpi-value">{c_chk}/{tot}</div><div class="kpi-label">{c_pct*100:.1f}%</div></div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# SIDEBAR ANALYTICS (all charts live here)
# -------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-card"><div class="sidebar-title">Analytics</div><div class="sidebar-sub">Live charts for the current view</div>', unsafe_allow_html=True)

    # Overall donut
    fig = plt.figure()
    sizes = [both_chk, max(tot - both_chk, 0)]
    labels = ["Completed (✓✓)", "Remaining"]
    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%", startangle=90)
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    fig.gca().add_artist(centre_circle)
    plt.title("Overall completion")
    st.pyplot(fig, use_container_width=True)

    # Role donuts
    fig = plt.figure()
    sizes = [m_chk, max(tot - m_chk, 0)]
    labels = ["Modelador ✓", "Remaining"]
    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%", startangle=90)
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    fig.gca().add_artist(centre_circle)
    plt.title("Modelador progress")
    st.pyplot(fig, use_container_width=True)

    fig = plt.figure()
    sizes = [c_chk, max(tot - c_chk, 0)]
    labels = ["Coordinador ✓", "Remaining"]
    plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%", startangle=90)
    centre_circle = plt.Circle((0,0),0.60,fc='white')
    fig.gca().add_artist(centre_circle)
    plt.title("Coordinador progress")
    st.pyplot(fig, use_container_width=True)

    # Per-section dataframe (current view)
    sec_df = (
        view
        .assign(
            both=lambda d: [
                1 if (st.session_state['checks'][rid]['MODELADOR']==1 and st.session_state['checks'][rid]['COORDINADOR']==1) else 0
                for rid in d['item_id']
            ]
        )
        .groupby(["section_code","section_title"], as_index=False)
        .agg(total=("item_id","count"), done=("both","sum"))
        .assign(pct=lambda d: (d["done"]/d["total"]*100).round(1))
        .sort_values(["section_code"])
    )

    # Leaderboard bar
    if not sec_df.empty:
        fig = plt.figure(figsize=(3.6, 2.2))
        plt.bar(sec_df["section_code"], sec_df["pct"])
        plt.title("By section (✓✓ %)")
        plt.ylabel("%")
        plt.ylim(0, 100)
        plt.xticks(rotation=90, fontsize=7)
        st.pyplot(fig, use_container_width=True)
    else:
        st.caption("No sections in view.")

    # Section picker + donut
    if not sec_df.empty:
        sel = st.selectbox(
            "Focus section",
            options=sec_df["section_code"].tolist(),
            format_func=lambda s: f"{s} — {sec_df.loc[sec_df['section_code']==s, 'section_title'].values[0]}",
        )
        row = sec_df.loc[sec_df["section_code"]==sel].iloc[0]
        st.markdown(
            f"<div class='smallnote'><b>{sel}</b> • {row['section_title']}<br>"
            f"{int(row['done'])} ✓✓ / {int(row['total'])} ({row['pct']}%)</div>",
            unsafe_allow_html=True
        )
        s_total = int(row["total"])
        s_done = int(row["done"])
        fig = plt.figure()
        sizes = [s_done, max(s_total - s_done, 0)]
        labels = ["Completed (✓✓)", "Remaining"]
        plt.pie(sizes, labels=labels, autopct=lambda p: f"{p:.1f}%", startangle=90)
        centre_circle = plt.Circle((0,0),0.60,fc='white')
        fig.gca().add_artist(centre_circle)
        plt.title(f"{sel} completion")
        st.pyplot(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close sidebar-card

# -------------------------
# LIVE BADGE (main)
# -------------------------
st.markdown('<div class="band"><span class="badge">Live — Interactive</span></div>', unsafe_allow_html=True)

# -------------------------
# CHECKLIST (main)
# -------------------------
st.markdown('<div class="band">', unsafe_allow_html=True)
for section_key, group in view.groupby(["Discipline","section_code","section_title"], sort=False):
    _, code, title = section_key
    st.markdown(f"<div class='section-header'>{code} — {title}</div>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns([.9, .6, .6])
    with b1: st.caption("Bulk toggle for this section:")
    with b2:
        if st.button(f"Modelador: All ✓ ({code})"):
            for _, r in group.iterrows():
                st.session_state['checks'][r['item_id']]['MODELADOR'] = 1
    with b3:
        if st.button(f"Coordinador: All ✓ ({code})"):
            for _, r in group.iterrows():
                st.session_state['checks'][r['item_id']]['COORDINADOR'] = 1

    for _, r in group.iterrows():
        cc1, cc2, cc3 = st.columns([8,2,2])
        with cc1:
            st.markdown(f"<div class='row'>{r['description']}</div>", unsafe_allow_html=True)
        with cc2:
            key_m = f"M_{r['item_id']}"
            st.session_state["checks"][r["item_id"]]["MODELADOR"] = st.checkbox(
                "Modelador", value=bool(st.session_state["checks"][r["item_id"]]["MODELADOR"]), key=key_m
            )
            st.session_state["checks"][r["item_id"]]["MODELADOR"] = int(st.session_state["checks"][r["item_id"]]["MODELADOR"])
        with cc3:
            key_c = f"C_{r['item_id']}"
            st.session_state["checks"][r["item_id"]]["COORDINADOR"] = st.checkbox(
                "Coordinador", value=bool(st.session_state["checks"][r["item_id"]]["COORDINADOR"]), key=key_c
            )
            st.session_state["checks"][r["item_id"]]["COORDINADOR"] = int(st.session_state["checks"][r["item_id"]]["COORDINADOR"])

    st.markdown("<hr style='border:none;height:1px;background:var(--stroke);opacity:.6;margin:8px 0;'/>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Export (main)
# -------------------------
def to_excel_bytes(items_df: pd.DataFrame, checks_state: dict) -> bytes:
    out_df = items_df.copy()
    out_df["MODELADOR"] = out_df["item_id"].apply(lambda i: int(checks_state[i]["MODELADOR"]))
    out_df["COORDINADOR"] = out_df["item_id"].apply(lambda i: int(checks_state[i]["COORDINADOR"]))

    rows, current_code = [], None
    for _, r in items_df.iterrows():
        if r["section_code"] != current_code:
            current_code = r["section_code"]
            rows.append([current_code, r["section_title"], "", ""])
        rows.append(["", r["description"],
                     int(checks_state[r["item_id"]]["MODELADOR"]),
                     int(checks_state[r["item_id"]]["COORDINADOR"])])

    flat_df = pd.DataFrame(rows, columns=["Code","Description","MODELADOR","COORDINADOR"])

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        out_df.to_excel(writer, index=False, sheet_name="Items")
        flat_df.to_excel(writer, index=False, sheet_name="Checklist")
    bio.seek(0)
    return bio.read()

st.markdown('<div class="band">', unsafe_allow_html=True)
excel_bytes = to_excel_bytes(items, st.session_state["checks"])
st.download_button(
    "⬇️ Export (Excel)",
    data=excel_bytes,
    file_name="QA_Checklist_Updated.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
st.markdown('</div>', unsafe_allow_html=True)
