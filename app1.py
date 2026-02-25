import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# CONFIG
# ════════════════════════════════════════════════════════════════
MODEL = "google/gemma-3-27b-it"
RESULTS_N = 5
# OpenRouter hinnad (USD per 1M tokenit) – uuenda vastavalt mudelile
PRICE_INPUT_PER_1M  = 0.04
PRICE_OUTPUT_PER_1M = 0.15

COLS_SHOW = ["aine_kood", 'nimi_et', 'nimi_en', 'eap', "semester", "oppeaste",
             "keel", "linn", "hindamisviis"]

# ANDMED JA MUDEL (cache)
# ════════════════════════════════════════════════════════════════
@st.cache_resource
def load_resources():
    embedder      = SentenceTransformer("BAAI/bge-m3")
    df            = pd.read_csv("andmed/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("andmed/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df

embedder, df, embeddings_df = load_resources()

# ABIFUNKTSIOONID
# ════════════════════════════════════════════════════════════════
def build_mask(source_df: pd.DataFrame, eristav, semester, aste, keel, asukoht) -> pd.Series:
    """Rakenda kõik filtrid etteantud DataFrame'ile. Tagastab bool-maski."""
    mask = pd.Series(True, index=source_df.index)

    if eristav == "A–F":
        mask &= source_df["hindamisviis"].str.contains("Eristav", case=False, na=False)
    elif eristav == "arvestatud/mittearvestatud":
        mask &= source_df["hindamisviis"].str.contains("Eristamata", case=False, na=False)

    if semester:
        mask &= source_df["semester"].isin(semester)

    if aste:
        mask &= source_df["oppeaste"].str.contains("|".join(aste), case=False, na=False)

    if keel:
        mask &= source_df["keel"].str.contains("|".join(keel), case=False, na=False)

    if asukoht != "Pole oluline":
        mask &= source_df["linn"].str.contains(asukoht, case=False, na=False)

    return mask


def semantic_search(filtered_df: pd.DataFrame, query: str, n: int) -> pd.DataFrame:
    """Lisab embeddings'id, arvutab sarnasuse ja tagastab top-n read."""
    merged = pd.merge(filtered_df, embeddings_df, on="unique_ID", how="left")
    merged = merged.dropna(subset=["embedding"])
    if merged.empty:
        return merged

    query_vec  = embedder.encode(query, normalize_embeddings=True)
    emb_matrix = np.stack(merged["embedding"].values)
    merged["score"] = cosine_similarity([query_vec], emb_matrix)[0]

    return merged.sort_values("score", ascending=False).head(n)


def format_context(top: pd.DataFrame) -> str:
    cols  = [c for c in COLS_SHOW if c in top.columns]
    table = top[cols].to_string(index=False)
    return (
        "Leitud sobivaimad kursused (sarnasuse järgi):\n\n"
        f"{table}\n\n"
        "Toetu eelkõige kirjeldusele, soovita parimaid sarnasusi."
    )

def filter_summary_text(eristav, semester, aste, keel, asukoht) -> str:
    parts = []
    if eristav != "Pole oluline":
        parts.append(f"hindamine: {eristav}")
    if semester:
        parts.append(f"semester: {', '.join(semester)}")
    if aste:
        parts.append(f"aste: {', '.join(aste)}")
    if keel:
        parts.append(f"keel: {', '.join(keel)}")
    if asukoht != "Pole oluline":
        parts.append(f"asukoht: {asukoht}")
    return ("Rakendatud filtrid: " + "; ".join(parts)) if parts else "Filtreid ei rakendatud."


def compute_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens  * PRICE_INPUT_PER_1M +
            output_tokens * PRICE_OUTPUT_PER_1M) / 1_000_000


# PEALKIRI
# ════════════════════════════════════════════════════════════════
st.title("🎓 AI Kursuse Nõustaja")
st.caption("RAG süsteem koos metaandmete filtreerimisega")

# KÜLGRIBA
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")

    st.divider()
    st.subheader("🔍 Filtrid")
    st.caption("Jäta tühjaks, kui filter pole oluline.")

    eristav_valik = st.radio(
        "Hindamine",
        ["Pole oluline", "A–F", "arvestatud/mittearvestatud"],
        index=0,
    )
    semester_valik = st.multiselect(
        "Semester", ["kevad", "sügis"],
        placeholder="Kõik semestrid",
    )
    aste_valik = st.multiselect(
        "Õppeaste",
        ["bakalaureuseõpe", "magistriõpe", "doktoriõpe",
         "rakenduskõrgharidusõpe", "integreeritud bakalaureuse- ja magistriõpe"],
        placeholder="Kõik astmed",
    )
    keel_valik = st.multiselect(
        "Õppekeel", ["eesti", "inglise", "vene", "muu"],
        placeholder="Kõik keeled",
    )
    asukoht_valik = st.selectbox(
        "Asukoht",
        ["Pole oluline", "Tartu", "Tallinn", "Narva", "Pärnu", "Viljandi"],
    )

    # ── Kursuste arv (uueneb iga filtrimuutusega) ──────────
    st.divider()
    live_mask = build_mask(df, eristav_valik, semester_valik,
                           aste_valik, keel_valik, asukoht_valik)
    n_live = int(live_mask.sum())
    st.metric("📚 Kursusi filtrite järgi", n_live)
    if n_live == 0:
        st.warning("Ükski kursus ei vasta praegustele filtritele.")

    # ── Kulu. Jooksev kokkuvõte ───────────────────────────────────
    st.divider()
    if "total_cost" not in st.session_state:
        st.session_state.total_cost   = 0.0
        st.session_state.total_input  = 0
        st.session_state.total_output = 0

    st.metric("💰 Jooksev kulu (USD)", f"${st.session_state.total_cost:.5f}")
    st.caption(
        f"Sisend: {st.session_state.total_input} tok  |  "
        f"Väljund: {st.session_state.total_output} tok"
    )
    if st.button("🗑️ Tühjenda vestlus"):
        st.session_state.messages     = []
        st.session_state.total_cost   = 0.0
        st.session_state.total_input  = 0
        st.session_state.total_output = 0
        st.rerun()

# VESTLUSAKEN
# ════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            msg = "Palun sisesta OpenRouter API võti külgribal."
            st.error(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.stop()

        with st.spinner("Otsin sobivaid kursusi..."):

            # 1. Filtreeri
            mask        = build_mask(df, eristav_valik, semester_valik,
                                     aste_valik, keel_valik, asukoht_valik)
            filtered_df = df[mask]

            # 2. Semantiline otsing
            if filtered_df.empty:
                context_text = "Sobivaid kursusi ei leitud – filtrid on liiga kitsad."
            else:
                top          = semantic_search(filtered_df, prompt, RESULTS_N)
                context_text = (format_context(top) if not top.empty
                                else "Embeddings puuduvad leitud kursustel.")

            # 3. Süsteemiprompt
            system_msg = {
                "role": "system",
                "content": (
                    "Oled abivalmis Eesti kõrgkoolide kursuste nõustaja. "
                    f"{filter_summary_text(eristav_valik, semester_valik, aste_valik, keel_valik, asukoht_valik)}\n\n"
                    f"{context_text}\n\n"
                    "Vasta eesti keeles. Too iga kursuse kohta välja: nimi, kood, maht (EAP), keel ja asukoht. "
                    "Kuva kursuse link markdown formaadis: [AINE_KOOD](https://ois2.ut.ee/#/courses/AINE_KOOD) "
                    "kus AINE_KOOD on täpne kood tabelist (nt MTAT.03.206). "
                    "Kui sobivaid kursusi ei leitud, soovita filtrite laiendamist."
                ),
            }

            # 4. LLM kutse
            client           = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            messages_to_send = [system_msg] + st.session_state.messages

            try:
                stream = client.chat.completions.create(
                    model=MODEL,
                    messages=messages_to_send,
                    temperature=0.6,
                    max_tokens=1800,
                    stream=True,
                    stream_options={"include_usage": True},
                )

                full_response = ""
                placeholder   = st.empty()

                for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        full_response += delta
                        placeholder.markdown(full_response + "▌")

                    # Viimane chunk sisaldab tokenite arvu
                    if hasattr(chunk, "usage") and chunk.usage:
                        in_tok  = chunk.usage.prompt_tokens
                        out_tok = chunk.usage.completion_tokens
                        st.session_state.total_input  += in_tok
                        st.session_state.total_output += out_tok
                        st.session_state.total_cost   += compute_cost(in_tok, out_tok)

                placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except Exception as e:
                err = f"Viga LLM-iga ühenduses: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
