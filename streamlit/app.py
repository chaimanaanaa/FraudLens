"""
FraudLens — Streamlit frontend.
Left panel: choose an image. Right panel: verdict returned by the n8n webhook.

Message content is attacker-controlled: it is always rendered through escaped HTML (st.html),
never through st.markdown, which would turn scam URLs into clickable links.
"""
import base64
import html
import os
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import requests
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps, UnidentifiedImageError

N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook/fraudlens")
REQUEST_TIMEOUT_SECONDS = 300
MAX_IMAGE_SIDE = 2000

VERDICTS = {
    "SCAM": {
        "color": "#C81E1E",
        "bg": "#FDF0F0",
        "icon": "🚨",
        "label": "Arnaque détectée",
        "advice": "Ne répondez pas, ne cliquez sur aucun lien et ne payez rien.",
    },
    "SUSPICIOUS": {
        "color": "#B45309",
        "bg": "#FEF6E7",
        "icon": "⚠️",
        "label": "Message suspect",
        "advice": "Par prudence, vérifiez auprès de l'organisme par un canal officiel avant d'agir.",
    },
    "LEGITIMATE": {
        "color": "#047857",
        "bg": "#ECFDF5",
        "icon": "✅",
        "label": "Message légitime",
        "advice": "Aucun signe d'arnaque détecté. Restez tout de même vigilant.",
    },
}

FRAUD_TYPES = {
    "phishing": "Hameçonnage (phishing)",
    "smishing": "SMS frauduleux (smishing)",
    "vishing": "Arnaque téléphonique (vishing)",
    "advance_fee": "Arnaque à l'avance de frais",
    "impersonation": "Usurpation d'identité",
    "other": "Autre fraude",
}

STEPS = [
    ("Lecture", "le texte est extrait de votre image"),
    ("Comparaison", "il est comparé à des arnaques connues"),
    ("Analyse", "une IA rend un verdict argumenté"),
    ("Vérification", "le verdict est contrôlé avant affichage"),
]

CSS = """
<style>
[data-testid="stHeader"] {display: none;}
.stApp {background: #F4F6F9;}
.block-container {max-width: 1240px; padding-top: 1.5rem; padding-bottom: 1rem;}
.fl-brand {display: flex; align-items: center; gap: 12px; margin-bottom: 4px;}
.fl-logo {font-size: 1.9rem; line-height: 1;}
.fl-name {font-size: 1.65rem; font-weight: 700; color: #111827; line-height: 1.2;}
.fl-tagline {font-size: .95rem; color: #4B5563;}

[data-testid="stHorizontalBlock"] > [data-testid="column"] > [data-testid="stVerticalBlockBorderWrapper"]
  > div > [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
  background: #FFFFFF; border-color: #E5E7EB; border-radius: 14px; box-shadow: 0 1px 3px rgba(17, 24, 39, .06);
}
@media (min-width: 641px) {
  [data-testid="column"]:has(.fl-left) {position: sticky; top: 1rem; align-self: flex-start;}
}
.fl-panel-title {font-size: 1.2rem; font-weight: 700; color: #111827; margin: 2px 0;}
.fl-panel-hint {font-size: .9rem; color: #4B5563;}

[data-testid="stFileUploaderDropzone"] {
  flex-direction: column; gap: 12px; padding: 26px 16px; background: #F8FAFF;
  border: 2px dashed #BFCDEB; border-radius: 12px; transition: background .15s, border-color .15s;
}
[data-testid="stFileUploaderDropzone"]:hover {background: #EEF3FF; border-color: #1D4ED8;}
[data-testid="stFileUploaderDropzoneInstructions"] {margin: 0;}
[data-testid="stFileUploaderDropzoneInstructions"] > span {display: none;}
[data-testid="stFileUploaderDropzoneInstructions"] > div {display: flex; flex-direction: column; align-items: center; gap: 2px;}
[data-testid="stFileUploaderDropzoneInstructions"] div span,
[data-testid="stFileUploaderDropzoneInstructions"] div small {font-size: 0;}
[data-testid="stFileUploaderDropzoneInstructions"] div span::before {
  content: "Glissez votre image ici"; font-size: 1rem; font-weight: 600; color: #111827;
}
[data-testid="stFileUploaderDropzoneInstructions"] div small::before {
  content: "JPG, PNG, WEBP ou BMP, 15 Mo maximum"; font-size: .8rem; color: #6B7280;
}
[data-testid="stFileUploaderDropzone"] button {font-size: 0;}
[data-testid="stFileUploaderDropzone"] button::before {content: "Choisir un fichier"; font-size: .95rem;}

[data-testid="stImage"] img {max-height: 340px; object-fit: contain; border-radius: 10px;}
[data-testid="StyledFullScreenButton"] {top: 6px; right: 6px;}
.stButton button {border-radius: 10px; font-weight: 600; transition: transform .1s, box-shadow .15s;}
.stButton button:hover:enabled {transform: translateY(-1px); box-shadow: 0 2px 6px rgba(17, 24, 39, .12);}
[data-testid="stExpander"] details {border-radius: 10px; border-color: #E5E7EB;}

.fl-hero {border-left: 6px solid; border-radius: 12px; padding: 16px 20px; animation: fl-rise .3s ease-out;}
.fl-hero h2 {margin: 0; padding: 0; font-size: 1.55rem; font-weight: 700;}
.fl-hero p {margin: 4px 0 12px; color: #374151;}
.fl-meter-label {display: flex; justify-content: space-between; gap: 12px; font-size: .875rem; color: #374151; margin-bottom: 4px;}
.fl-meter {height: 8px; border-radius: 4px; background: rgba(0, 0, 0, .08); overflow: hidden;}
.fl-meter div {height: 100%; border-radius: 4px; animation: fl-fill .7s ease-out;}
.fl-checked {margin-top: 10px; font-size: .8rem; color: #4B5563;}
.fl-section {font-size: 1rem; font-weight: 700; color: #111827; margin: 4px 0 6px;}
.fl-list {margin: 0 0 4px; padding-left: 1.2rem; color: #1F2937;}
.fl-list li {margin-bottom: 6px; overflow-wrap: anywhere;}
.fl-todo {list-style: none; counter-reset: todo; margin: 0 0 4px; padding: 0; color: #1F2937;}
.fl-todo li {counter-increment: todo; position: relative; padding-left: 36px; margin-bottom: 10px; min-height: 26px; overflow-wrap: anywhere;}
.fl-todo li::before {
  content: counter(todo); position: absolute; left: 0; top: -1px; width: 26px; height: 26px; border-radius: 50%;
  background: var(--fl-accent); color: #FFFFFF; font-size: .8rem; font-weight: 700; line-height: 26px; text-align: center;
}
.fl-box {background: #F3F4F6; border-radius: 8px; padding: 12px 14px; white-space: pre-wrap;
         overflow-wrap: anywhere; font-size: .9rem; color: #1F2937; margin-bottom: 8px;}
.fl-pill {display: inline-block; border-radius: 999px; padding: 1px 10px; font-size: .75rem; font-weight: 600;}
.fl-pill-spam {background: #FDE2E2; color: #9B1C1C;}
.fl-pill-ham {background: #D1FAE5; color: #065F46;}
.fl-pill-unknown {background: #E5E7EB; color: #374151;}
.fl-score {font-size: .8rem; color: #6B7280; margin-left: 6px;}
.fl-steps {display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0 4px;}
.fl-step {display: flex; gap: 12px; background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 10px; padding: 12px 14px;
          transition: border-color .15s, background .15s;}
.fl-step:hover {border-color: #BFCDEB; background: #F8FAFF;}
.fl-step i {flex: none; width: 26px; height: 26px; border-radius: 50%; background: #1D4ED8; color: #FFFFFF;
            font-style: normal; font-size: .8rem; font-weight: 700; line-height: 26px; text-align: center;}
.fl-step b {display: block; margin-bottom: 2px; color: #111827;}
.fl-step span {font-size: .875rem; color: #4B5563;}
@keyframes fl-rise {from {opacity: 0; transform: translateY(6px);} to {opacity: 1; transform: none;}}
@keyframes fl-fill {from {width: 0;}}
@media (max-width: 640px) {.fl-steps {grid-template-columns: 1fr;}}
@media (prefers-reduced-motion: reduce) {.fl-hero, .fl-meter div {animation: none;} .stButton button {transition: none;}}
</style>
"""

SCROLL_TO_VERDICT = """
<script>
setTimeout(() => {
  const parent = window.parent;
  const anchor = parent.document.querySelector("[data-testid='column']:not(:has(.fl-left)) .fl-anchor");
  if (anchor && parent.innerWidth <= 640) anchor.scrollIntoView({behavior: "smooth", block: "start"});
}, 200);
</script>
"""


def init_state() -> None:
    defaults = {
        "source": None, "uploader_key": 0, "running": False, "result": None, "error": None, "scroll_to_verdict": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_analysis() -> None:
    st.session_state.result = None
    st.session_state.error = None


def select_source(name: str, data: bytes) -> None:
    st.session_state.source = {"name": name, "data": data}
    st.session_state.uploader_key += 1
    clear_analysis()


def accept_upload(widget_key: str) -> None:
    uploaded = st.session_state.get(widget_key)
    if uploaded is not None:
        select_source(uploaded.name, uploaded.getvalue())


def start_over() -> None:
    st.session_state.source = None
    clear_analysis()


def start_analysis() -> None:
    clear_analysis()
    st.session_state.running = True


def load_image(data: bytes) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return image


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92, subsampling=0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_fraudlens(image_b64: str, filename: str) -> dict:
    payload = {"image_base64": image_b64, "filename": filename}
    response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def analyze_with_timer(image: Image.Image, filename: str) -> dict:
    """Runs the blocking webhook call in a worker so the elapsed time stays visible."""
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(call_fraudlens, image_to_base64(image), filename)
        started = time.monotonic()
        with st.spinner("Analyse en cours..."):
            timer = st.empty()
            while not future.done():
                elapsed = int(time.monotonic() - started)
                timer.caption(f"{elapsed} s écoulées. Une analyse prend en général de 30 s à 2 minutes.")
                time.sleep(0.5)
        return future.result()
    finally:
        executor.shutdown(wait=False)


def run_analysis(image: Image.Image, filename: str) -> None:
    render_panel_title("Analyse de votre message")
    render_steps()
    try:
        result = analyze_with_timer(image, filename)
        if "verdict" not in result:
            raise ValueError(f"réponse inattendue : {str(result)[:200]}")
        st.session_state.result = result
    except requests.exceptions.ConnectionError:
        st.session_state.error = (
            "Impossible de joindre le service d'analyse.",
            "Vérifiez que les services Docker sont démarrés : docker compose up -d",
        )
    except requests.exceptions.Timeout:
        st.session_state.error = (
            "L'analyse a dépassé le délai de 5 minutes.",
            "Le service Gemini est peut-être saturé. Réessayez dans quelques minutes.",
        )
    except requests.exceptions.HTTPError as exc:
        st.session_state.error = (
            f"Le service d'analyse a renvoyé une erreur ({exc.response.status_code}).",
            "Le quota Gemini est peut-être atteint, ou le workflow n8n inactif. "
            "Détail dans : docker compose logs backend",
        )
    except Exception as exc:
        st.session_state.error = ("Une erreur inattendue s'est produite.", str(exc))
    finally:
        st.session_state.running = False
        st.session_state.scroll_to_verdict = True


def bullets(items) -> str:
    rows = "".join(f"<li>{html.escape(str(item))}</li>" for item in items)
    return f'<ul class="fl-list">{rows}</ul>'


def render_panel_title(title: str, hint: str = "") -> None:
    hint_html = f'<div class="fl-panel-hint">{hint}</div>' if hint else ""
    st.html(f'<div class="fl-anchor"></div><div class="fl-panel-title">{title}</div>{hint_html}')


def render_steps() -> None:
    cards = "".join(
        f'<div class="fl-step"><i>{index}</i><div><b>{title}</b><span>{detail}</span></div></div>'
        for index, (title, detail) in enumerate(STEPS, 1)
    )
    st.html(f'<div class="fl-steps">{cards}</div>')


def render_empty() -> None:
    render_panel_title(
        "Le verdict s'affichera ici",
        "Choisissez une image à gauche, puis lancez l'analyse. Voici ce que fait FraudLens :",
    )
    render_steps()


def render_error() -> None:
    title, detail = st.session_state.error
    render_panel_title("L'analyse n'a pas abouti", "Votre image est conservée : vous pouvez réessayer.")
    st.error(title)
    st.html(f'<div class="fl-box">{html.escape(detail)}</div>')


def render_no_text() -> None:
    render_panel_title("Aucun texte détecté")
    st.warning(
        "Impossible de lire un message sur cette image, donc aucun verdict n'est possible. "
        "Essayez une photo plus nette, bien cadrée sur le texte."
    )


def render_hero(result: dict) -> dict:
    verdict = str(result.get("verdict", "SUSPICIOUS")).upper()
    cfg = VERDICTS.get(verdict, VERDICTS["SUSPICIOUS"])
    confidence = max(0.0, min(float(result.get("confidence_score") or 0.0), 1.0))
    fraud_type = result.get("fraud_type")
    type_label = html.escape(FRAUD_TYPES.get(fraud_type, str(fraud_type))) if fraud_type else ""
    checked = '<div class="fl-checked">✓ Verdict contrôlé par l\'agent de validation</div>' if result.get("is_valid") else ""
    st.html(
        '<div class="fl-anchor"></div>'
        f'<div class="fl-hero" style="background:{cfg["bg"]};border-color:{cfg["color"]}">'
        f'<h2 style="color:{cfg["color"]}">{cfg["icon"]} {cfg["label"]}</h2>'
        f'<p>{cfg["advice"]}</p>'
        f'<div class="fl-meter-label"><span>Confiance : <b>{confidence:.0%}</b></span><span>{type_label}</span></div>'
        f'<div class="fl-meter"><div style="width:{confidence:.0%};background:{cfg["color"]}"></div></div>'
        f"{checked}</div>"
    )
    return cfg


def render_sources(result: dict) -> None:
    st.caption("Les messages de notre base les plus proches du vôtre, et leur nature.")
    passages = result.get("passages") or []
    if not passages:
        st.write("Aucun message similaire trouvé.")
    labels = {"spam": ("ARNAQUE", "fl-pill-spam"), "ham": ("LÉGITIME", "fl-pill-ham")}
    for passage in passages:
        label, pill = labels.get(passage.get("label"), ("INCONNU", "fl-pill-unknown"))
        excerpt = html.escape(str(passage.get("text", ""))[:220])
        score = float(passage.get("score") or 0.0)
        st.html(
            f'<span class="fl-pill {pill}">{label}</span>'
            f'<span class="fl-score">{score:.0%} de similarité</span>'
            f'<div class="fl-box">{excerpt}</div>'
        )
    cited = result.get("cited_sources") or []
    if cited:
        st.html(f'<div class="fl-section">Passages cités par l\'agent d\'analyse</div>{bullets(cited)}')


def render_result(result: dict) -> None:
    if not str(result.get("extracted_text", "")).strip():
        render_no_text()
        return

    cfg = render_hero(result)
    if result.get("regenerated"):
        st.info("La première analyse n'a pas passé les contrôles de qualité : celle-ci a été régénérée.")

    st.html(f'<div class="fl-section">Pourquoi ce verdict ?</div>{bullets(result.get("reasons") or [])}')
    actions = "".join(f"<li>{html.escape(str(a))}</li>" for a in result.get("recommended_actions") or [])
    st.html(
        '<div class="fl-section">Que faire maintenant ?</div>'
        f'<ol class="fl-todo" style="--fl-accent:{cfg["color"]}">{actions}</ol>'
    )
    with st.expander("Texte lu sur l'image"):
        st.caption("Vérifiez que le texte a été bien lu : le verdict repose dessus.")
        st.html(f'<div class="fl-box">{html.escape(str(result["extracted_text"]))}</div>')
    with st.expander("Messages similaires dans notre base"):
        render_sources(result)


def render_picker() -> None:
    render_panel_title("Votre message", "Photo ou capture d'écran d'un SMS, d'un email ou d'un courrier.")
    widget_key = f"uploader_{st.session_state.uploader_key}"
    st.file_uploader(
        "Importer une image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key=widget_key,
        label_visibility="collapsed",
        on_change=accept_upload,
        args=(widget_key,),
    )


def render_actions(image_ok: bool) -> None:
    running = st.session_state.running
    if st.session_state.result and not running:
        st.button("Analyser une autre image", type="primary", on_click=start_over, use_container_width=True)
        return
    if image_ok:
        label = "Analyse en cours..." if running else "Réessayer" if st.session_state.error else "Analyser ce message"
        st.button(label, type="primary", on_click=start_analysis, disabled=running, use_container_width=True)
    st.button("Changer d'image", on_click=start_over, disabled=running, use_container_width=True)


def render_preview():
    """Returns the decoded image, or None if the file is not a readable image."""
    render_panel_title("Votre message")
    try:
        image = load_image(st.session_state.source["data"])
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error("Ce fichier n'est pas une image lisible. Essayez un autre fichier.")
        image = None
    if image is not None:
        st.image(image, use_column_width=True)
    render_actions(image is not None)
    return image


st.set_page_config(page_title="FraudLens", page_icon="🔍", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
init_state()

st.html(
    '<div class="fl-brand"><span class="fl-logo">🔍</span><div>'
    '<div class="fl-name">FraudLens</div>'
    '<div class="fl-tagline">Un message vous paraît louche ? Photographiez-le, FraudLens vous dit quoi en penser.</div>'
    "</div></div>"
)

left, right = st.columns([5, 7], gap="large")

with left:
    with st.container(border=True):
        st.html('<div class="fl-left"></div>')
        image = render_preview() if st.session_state.source else render_picker()

with right:
    with st.container(border=True):
        if st.session_state.running:
            run_analysis(image, st.session_state.source["name"])
            st.rerun()
        elif st.session_state.error:
            render_error()
        elif st.session_state.result:
            render_result(st.session_state.result)
        else:
            render_empty()
        if st.session_state.scroll_to_verdict:
            st.session_state.scroll_to_verdict = False
            components.html(SCROLL_TO_VERDICT, height=0)

st.caption("FraudLens donne un avis indicatif. En cas de doute, contactez directement l'organisme concerné.")
