# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 10:03:38 2026
Author: V. Moonsamy
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError, ImageOps

# ---------------------------------------------------------------------------
# Page config MUST be first
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Researcher Profile & Urban Development Data Explorer",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def about_metric(title: str, body_md: str, expanded: bool = True):
    """Renders a consistent expander with a metric explainer."""
    with st.expander(f"About this metric: {title}", expanded=expanded):
        st.markdown(body_md)

@st.cache_data(show_spinner=False)
def list_images(folder: Path):
    """Return image files in a folder, filtered by common extensions."""
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def make_uniform_tile(img: Image.Image, box_size: tuple[int, int], bg_color=(245, 245, 245)) -> Image.Image:
    """
    Letterbox: scale to fit within box_size, keep aspect, center on background (no cropping).
    """
    w_box, h_box = box_size
    img = img.convert("RGB")
    img_fit = ImageOps.contain(img, (w_box, h_box), method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w_box, h_box), color=bg_color)
    x = (w_box - img_fit.width) // 2
    y = (h_box - img_fit.height) // 2
    canvas.paste(img_fit, (x, y))
    return canvas

def render_image_grid(folder: Path,
                      n_cols: int = 3,
                      caption_from_name: bool = True,
                      max_images: int = 30,
                      tile_size: tuple[int, int] = (360, 240),
                      bg_color=(245, 245, 245)):
    """
    Render images from `folder` as uniformly-sized tiles.
    - tile_size: (width_px, height_px)
    - bg_color: RGB tuple used for letterboxing
    """
    files = list_images(folder)
    if not files:
        st.info(f"No images found in `{folder}` or folder not present.")
        return

    files = files[:max_images]
    cols = st.columns(n_cols)

    for i, img_path in enumerate(files):
        with cols[i % n_cols]:
            try:
                with Image.open(str(img_path)) as im:
                    tile = make_uniform_tile(im, box_size=tile_size, bg_color=bg_color)
                cap = img_path.stem.replace("_", " ").title() if caption_from_name else None
                # Use explicit width so column alignment is consistent
                st.image(tile, caption=cap, width=tile_size[0], use_container_width=False)
            except UnidentifiedImageError:
                st.warning(f"Could not load image: {img_path.name}")
            except Exception as ex:
                st.error(f"Error processing {img_path.name}: {ex}")

@st.cache_data(show_spinner=False)
def list_pdfs(folder: Path):
    if folder.exists() and folder.is_dir():
        return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])
    return []

def show_pdf_iframe(url: str, height: int = 900):
    """
    Attempt to render a cross-origin PDF via <iframe>.
    May be blocked by the remote host's X-Frame-Options or CSP headers.
    """
    pdf_url = f"{url}#toolbar=1&navpanes=0&scrollbar=1"
    html = f"""
    <div style="height:{height}px; border:1px solid #ddd;">
      <iframe
        src="{pdf_url}"
        width="100%"
        height="{height}px"
        style="border:0;"
      ></iframe>
    </div>
    """
    components.html(html, height=height + 20, scrolling=False)

def show_pdf_base64(pdf_bytes: bytes, height: int = 900):
    """
    Inline, same-origin render via base64 data URL (works even when iframe is blocked).
    """
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    src = f"data:application/pdf;base64,{b64}#toolbar=1&navpanes=0&scrollbar=1"
    html = f"""
    <div style="height:{height}px; border:1px solid #ddd;">
      <embed src="{src}" type="application/pdf" width="100%" height="{height}px" />
    </div>
    """
    components.html(html, height=height + 20, scrolling=False)

@st.cache_data(show_spinner=False)
def fetch_pdf_bytes(url: str, timeout: int = 25) -> bytes | None:
    """
    Fetch a PDF from a public URL (server-side) and return its bytes.
    Cached to avoid repeated downloads.
    """
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

def render_pdf_with_fallback(url: str, height: int = 900, max_base64_mb: int = 12):
    """
    Try iframe first (fast path). If the host blocks embedding, let the user
    expand a fallback that fetches bytes server-side and shows a base64 <embed>.
    """
    # 1) Try iframe (may be blocked; if blocked, the frame shows a browser message)
    show_pdf_iframe(url, height=height)

    # 2) Optional inline fallback (user-triggered)
    with st.expander("If the viewer is blocked, click to try inline fallback"):
        pdf_bytes = fetch_pdf_bytes(url)
        if pdf_bytes is None:
            st.error("Could not fetch PDF for inline fallback (network error or blocked).")
            return

        size_mb = len(pdf_bytes) / (1024 * 1024)
        st.caption(f"Fetched PDF: {size_mb:.2f} MB")

        if size_mb <= max_base64_mb:
            show_pdf_base64(pdf_bytes, height=height)
            # Also offer direct download from the fetched bytes
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=url.split("/")[-1] or "document.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        else:
            st.warning(
                f"This file is {size_mb:.2f} MB which exceeds the {max_base64_mb} MB inline fallback threshold. "
                "Please use **Open in new tab** for best performance."
            )

def download_button_for_df(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )

def add_disclaimer():
    st.markdown(
        """
        ---
        **Disclaimer:**  
        The figures depicted in these tables and graphs are *notional* and were created solely
        for the purpose of completing an assignment for the **CSS6 Coding Summer School**.
        They do not represent actual City of Cape Town data.
        """
    )

# ---------------------------------------------------------------------------
# Demo datasets (tidied)
# ---------------------------------------------------------------------------
caprates_data = pd.DataFrame({
    "Year": [2019, 2020, 2021, 2022, 2023, 2024],
    "Capitalization Rate (%)": [8.7, 12.0, 10.9, 10.4, 11.5, 8.9],
}).sort_values("Year")

vacancyrate_data = pd.DataFrame({
    "Year": [2019, 2020, 2021, 2022, 2023, 2024],
    "Vacancy Rate (%)": [12.0, 29.0, 28.0, 21.0, 18.5, 15.0],
}).sort_values("Year")

floorarea_data = pd.DataFrame({
    "Business Precinct": ["Foreshore", "Bellville", "Claremont", "Century City", "Strand CBD"],
    "Rand per m² (ZAR)": [100, 50, 89, 111, 72],
    "Total Floor Area (m²)": [650_000, 200_000, 550_000, 600_000, 150_000],
    "As Of": pd.to_datetime("2024-01-01"),
})

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["Researcher Profile", "Branch Profile", "Publications", "Property Market Trends", "Contact Information"]
)

# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
if menu == "Researcher Profile":
    st.title("Researcher Profile")
    st.sidebar.header("Profile Options")

    # Optionally editable profile fields (persist in session)
    if "profile" not in st.session_state:
        st.session_state.profile = {
            "name": "Vernon Moonsamy",
            "field": "Urban Planning and Property Economics",
            "organisation": "City of Cape Town",
        }

    with st.sidebar:
        st.text_input("Name", key=("profile", "name"))
        st.text_input("Field of Research", key=("profile", "field"))
        st.text_input("Organisation", key=("profile", "organisation"))

    name = st.session_state.profile["name"]
    field = st.session_state.profile["field"]
    organisation = st.session_state.profile["organisation"]

    st.write(f"**Name:** {name}")
    st.write(f"**Field of Research:** {field}")
    st.write(f"**Organisation:** {organisation}")

    # Banner with graceful fallback
    external_banner = "https://everythingproperty.co.za/wp-content/uploads/2024/08/Cape-Town-property-market-mid-year-snapshot.jpg"
    try:
        st.image(external_banner, caption="Cape Town's Property Market (EverythingProperty)", use_container_width=True)
    except Exception:
        st.info("Banner unavailable. Add a local banner image to your images folder and reference it here.")

elif menu == "Branch Profile":
    st.title("Branch Profile")
    st.sidebar.header("Branch Profile Options")

    name = "Spatial Targeting and Mechanisms"
    department = "Urban Planning and Design"
    organisation = "City of Cape Town"

    st.write(f"**Name:** {name}")
    st.write(f"**Department:** {department}")
    st.write(f"**Organisation:** {organisation}")

    st.subheader("Branch Summary")
    st.markdown("""
The **Spatial Targeting & Mechanisms** branch plays a pivotal role within the Urban Planning and Design environment.
Its core focus areas include:
- **Spatial Targeting:** Identifying priority development areas that can unlock economic growth, improve service delivery, and guide public investment strategically.
- **Mechanisms Development:** Designing tools, frameworks, and methodologies that enable effective implementation of spatial strategies across the metro.
- **Urban & District Planning Support:** Providing analytical and spatial insights that assist Metro Planning, District Planning, and Urban Design units in evidence‑based decision‑making.
- **Data Analysis & Research:** Applying data‑driven approaches, spatial analytics, and research methods to understand development patterns, trends, and opportunities.
- **Interdepartmental Collaboration:** Working closely with external departments to align spatial initiatives with infrastructure planning, economic development, and service delivery priorities.

Overall, the branch enables more **coordinated**, **strategic**, and **impact‑oriented** urban planning through targeted interventions and robust spatial analysis.
""")
    # Optional local logo (only if present) — point to your new Images folder by default
     # ---- Logo (Direct Image URL) ----
    logo_url = "https://github.com/vernonmoon1331-oss/css_streamlit_2026_assignment/blob/main/CCT%20Logo.jpg?raw=true"
    st.markdown("### Logo")
    st.image(logo_url, caption="Spatial Targeting & Mechanisms", use_container_width=False)

elif menu == "Publications":
    st.title("Publications")
    st.sidebar.header("PDF Library (Public URLs)")

    # Viewer height control
    viewer_height = st.sidebar.slider("Viewer height (px)", 500, 1400, 900, step=50)

    # --- Public PDFs (extend as needed) ---
    pdf_items = [
        {
            "title": "Inclusionary Housing Overlay Zone: Overview (Technical Presentation)",
            "url": "https://resource.capetown.gov.za/documentcentre/Documents/Graphics%20and%20educational%20material/IOZ_Overview_Technical_Presentation.pdf",
        },
        {
            "title": "Table Bay DSDF/EMF Vol. 3",
            "url": "https://resource.capetown.gov.za/documentcentre/Documents/City%20strategies%2c%20plans%20and%20frameworks/Table_Bay_DSDF_EMF_Vol3.pdf",
        },
        {
            "title": "NHRA Pamphlet",
            "url": "https://resource.capetown.gov.za/documentcentre/Documents/Graphics%20and%20educational%20material/NHRA_Phamplet.pdf",
        },
    ]

    # Choose the document first so `selected` is defined for both columns
    titles = [p["title"] for p in pdf_items]
    selected_title = st.selectbox("Choose a document", titles, index=0)
    selected = next((p for p in pdf_items if p["title"] == selected_title), None)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("Library")
        st.caption(
            "If your browser blocks the inline viewer due to site settings, use **Open in new tab** "
            "or the **Inline fallback** on the right."
        )
        if selected:
            # Always-works fallback
            try:
                st.link_button("Open in new tab", selected["url"], use_container_width=True)
            except Exception:
                # Older Streamlit versions: provide a clickable markdown link instead
                st.markdown(f"{selected['url']}")
            with st.expander("Show source URL"):
                st.code(selected["url"], language="text")

    with right:
        st.subheader("PDF Viewer")
        if selected:
            render_pdf_with_fallback(selected["url"], height=viewer_height)
        else:
            st.info("Select a document on the left to view it.")

elif menu == "Property Market Trends":
    st.title("Property Market Trends")
    st.sidebar.header("Data Selection")

    data_option = st.sidebar.selectbox(
        "Choose a dataset to explore",
        ["Capitalization Rates", "Vacancy Rates", "Floor Area"]
    )

    # ------- Capitalization Rates -------
    if data_option == "Capitalization Rates":
        st.subheader("Cape Town Office Capitalization Rate Trends")
        about_metric(
            "Capitalization Rate (Cap Rate)",
            r"""
**Definition:** The capitalization rate (cap rate) is the ratio of a property's **net operating income (NOI)** to its **market value** (or purchase price).

**Formula:** Cap Rate = NOI / Market Value.

**Interpretation:** Higher cap rates generally imply **higher expected returns** and **higher perceived risk** (lower pricing), while lower cap rates imply **lower expected returns** and often **tighter pricing**—*all else equal*.
""",
            expanded=True
        )

        df = caprates_data.sort_values("Year")
        st.dataframe(df, use_container_width=True)

        min_rate = float(np.floor(df["Capitalization Rate (%)"].min()))
        max_rate = float(np.ceil(df["Capitalization Rate (%)"].max()))
        caprate_filter = st.slider(
            "Filter by Cap Rate (%)",
            min_value=min_rate,
            max_value=max_rate,
            value=(min_rate, max_rate),
            help="Cap Rate = NOI ÷ Market Value × 100%. Higher = higher implied return/risk; lower = tighter pricing."
        )

        filtered = df[df["Capitalization Rate (%)"].between(caprate_filter[0], caprate_filter[1])]
        st.write(f"Filtered Results for Capitalization Rates {caprate_filter}:")
        st.dataframe(filtered, use_container_width=True)

        # --- Cast Year to string to avoid thousands separators on x-axis
        tmp = filtered.copy()
        tmp["Year"] = tmp["Year"].astype(str)
        if not tmp.empty:
            st.line_chart(tmp.set_index("Year")["Capitalization Rate (%)"], height=240)

        # Disclaimer at bottom of this section
        add_disclaimer()

    # ------- Vacancy Rates -------
    elif data_option == "Vacancy Rates":
        st.subheader("Cape Town Office Vacancy Rate Trends")
        about_metric(
            "Vacancy Rate",
            r"""
**Definition:** Vacancy rate is the percentage of rentable space that is **unoccupied and available** at a given time (or over a period).

**Formula:** Vacancy Rate = Vacant Space / Total Rentable Space.

**Interpretation:** Higher vacancy rates generally indicate **weaker demand** or **excess supply**; lower vacancy rates indicate **tighter markets**, typically supporting stronger rental growth—*all else equal*.
""",
            expanded=True
        )

        df = vacancyrate_data.sort_values("Year")
        st.dataframe(df, use_container_width=True)

        min_vac = float(np.floor(df["Vacancy Rate (%)"].min()))
        max_vac = float(np.ceil(df["Vacancy Rate (%)"].max()))
        vacancy_filter = st.slider(
            "Filter by Vacancy Rate (%)",
            min_value=min_vac,
            max_value=max_vac,
            value=(min_vac, max_vac),
            help="Vacancy Rate = Vacant ÷ Total Rentable × 100%. Higher = softer demand; lower = tighter market."
        )

        filtered = df[df["Vacancy Rate (%)"].between(vacancy_filter[0], vacancy_filter[1])]
        st.write(f"Filtered Results for Vacancy Rate {vacancy_filter}:")
        st.dataframe(filtered, use_container_width=True)

        # --- Cast Year to string to avoid thousands separators on x-axis
        tmp = filtered.copy()
        tmp["Year"] = tmp["Year"].astype(str)
        if not tmp.empty:
            st.line_chart(tmp.set_index("Year")["Vacancy Rate (%)"], height=240)

        # Disclaimer at bottom of this section
        add_disclaimer()

    # ------- Floor Area -------
    elif data_option == "Floor Area":
        st.subheader("Rand per m² and Total Office Floor Area by Precinct")
        about_metric(
            "Rand per m² & Total Floor Area",
            """
**Rand per m² (ZAR):** A rental pricing benchmark indicating how much is paid (or asked) **per square metre** of space—useful for **relative pricing** across precincts or buildings.

**Total Floor Area (m²):** The aggregate **stock/scale** of space in the precinct, indicating **market depth**, potential **absorption capacity**, and concentration of activity.

**How to use them together for benchmarking:**
- **Price vs. Scale:** High **Rand/m²** + large **Total Floor Area** → established, deep markets with pricing power.
- **Emerging Value:** Moderate **Rand/m²** + growing **Total Floor Area** → transitioning nodes where demand is building.
- **Affordability:** Lower **Rand/m²** → cost‑effective locations; confirm access/amenities still meet requirements.
- **Prioritization:** Filter by **Rand/m²** for budget bands, then by **Total Floor Area** to ensure sufficient scale for expansion.
""",
            expanded=True
        )

        df = floorarea_data.copy()
        st.dataframe(df, use_container_width=True)

        rand_min = int(np.floor(df["Rand per m² (ZAR)"].min()))
        rand_max = int(np.ceil(df["Rand per m² (ZAR)"].max()))
        rand_sqm_filter = st.slider(
            "Filter by Rand per m² (ZAR)",
            min_value=rand_min,
            max_value=rand_max,
            value=(rand_min, rand_max),
            help="Use to match affordability bands across precincts (ZAR per m²)."
        )

        fa_min = int(np.floor(df["Total Floor Area (m²)"].min() / 50_000) * 50_000)
        fa_max = int(np.ceil(df["Total Floor Area (m²)"].max() / 50_000) * 50_000)
        total_floorarea_filter = st.slider(
            "Filter by Total Floor Area (m²)",
            min_value=fa_min,
            max_value=fa_max,
            value=(fa_min, fa_max),
            step=50_000,
            help="Screen for precincts with enough market depth/stock for current and future needs."
        )

        filtered = df[
            df["Rand per m² (ZAR)"].between(rand_sqm_filter[0], rand_sqm_filter[1])
            & df["Total Floor Area (m²)"].between(total_floorarea_filter[0], total_floorarea_filter[1])
        ]
        st.write(
            f"Filtered Results for Rand per m² {rand_sqm_filter} and Total Floor Area {total_floorarea_filter}:"
        )
        st.dataframe(filtered, use_container_width=True)

        if not filtered.empty:
            st.scatter_chart(
                filtered.rename(columns={
                    "Rand per m² (ZAR)": "Rand per m²",
                    "Total Floor Area (m²)": "Total Floor Area"
                }),
                x="Total Floor Area",
                y="Rand per m²",
                color="Business Precinct",
                height=300
            )

        # Disclaimer at bottom of this section
        add_disclaimer()

elif menu == "Contact Information":
    st.header("Contact Information")
    email = "Vernon.Moonsamy@capetown.gov.za"
    telephone = "021 123 4564"
    st.write(f"You can reach me at {email}.")
    st.write(f"Telephone: {telephone}")

    # Optional local logo (only if present) — point to your new Images folder by default
     # ---- Logo (Direct Image URL) ----
    logo_url = "https://github.com/vernonmoon1331-oss/css_streamlit_2026_assignment/blob/main/ST%26M%20Logo.jpeg"

    st.markdown("### Logo")
    st.image(logo_url, caption="Spatial Targeting & Mechanisms", use_container_width=False)



