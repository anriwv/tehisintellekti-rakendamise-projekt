import numpy as np
import pandas as pd
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    COLS_SHOW,
    DATA_CSV,
    DATA_EMB,
    MODEL,
    PRICE_INPUT_PER_1M,
    PRICE_OUTPUT_PER_1M,
    RESULTS_N,
)


# ── Resource loading (cached)

@st.cache_resource
def load_resources():
    """Load and cache the sentence embedder and both dataframes."""
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv(DATA_CSV)
    embeddings_df = pd.read_pickle(DATA_EMB)
    return embedder, df, embeddings_df


# ── Filtering

def build_mask(
    source_df: pd.DataFrame, eristav, semester, aste, keel, asukoht
) -> pd.Series:
    """Apply all sidebar filters and return a boolean mask."""
    mask = pd.Series(True, index=source_df.index)

    if eristav == "A–F":
        mask &= source_df["hindamisviis"].str.contains("Eristav", case=False, na=False)
    elif eristav == "arvestatud/mittearvestatud":
        mask &= source_df["hindamisviis"].str.contains(
            "Eristamata", case=False, na=False
        )

    if semester:
        mask &= source_df["semester"].isin(semester)

    if aste:
        mask &= source_df["oppeaste"].str.contains(
            "|".join(aste), case=False, na=False
        )

    if keel:
        mask &= source_df["keel"].str.contains("|".join(keel), case=False, na=False)

    if asukoht != "Pole oluline":
        mask &= source_df["linn"].str.contains(asukoht, case=False, na=False)

    return mask


def filter_summary_text(eristav, semester, aste, keel, asukoht) -> str:
    """Return a human-readable string summarising the active filters."""
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
    return (
        ("Rakendatud filtrid: " + "; ".join(parts))
        if parts
        else "Filtreid ei rakendatud."
    )


# ── Semantic search

def semantic_search(
    filtered_df: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    embedder: SentenceTransformer,
    query: str,
    n: int = RESULTS_N,
) -> pd.DataFrame:
    """Merge embeddings, compute cosine similarity, return top-n rows."""
    merged = pd.merge(filtered_df, embeddings_df, on="unique_ID", how="left")
    merged = merged.dropna(subset=["embedding"])
    if merged.empty:
        return merged

    query_vec = embedder.encode(query, normalize_embeddings=True)
    emb_matrix = np.stack(merged["embedding"].values)
    merged["score"] = cosine_similarity([query_vec], emb_matrix)[0]

    return merged.sort_values("score", ascending=False).head(n)


def format_context(top: pd.DataFrame) -> str:
    """Format the top courses as a string to be injected into the system prompt."""
    cols = [c for c in COLS_SHOW if c in top.columns]
    table = top[cols].to_string(index=False)
    return (
        "Leitud sobivaimad kursused (sarnasuse järgi):\n\n"
        f"{table}\n\n"
        "Toetu eelkõige kirjeldusele, soovita parimaid sarnasusi."
    )


# ── Cost calculation

def compute_cost(input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost for a given token pair."""
    return (
        input_tokens * PRICE_INPUT_PER_1M + output_tokens * PRICE_OUTPUT_PER_1M
    ) / 1_000_000


# ── LLM call

def run_llm_stream(
    api_key: str,
    messages: list,
    filter_text: str,
    context_text: str,
):
    """
    Stream a response from the LLM.

    Yields str chunks as they arrive, and finally yields a dict
    {"usage": {"input": int, "output": int}} when the stream ends.
    """
    system_msg = {
        "role": "system",
        "content": (
            "Oled abivalmis Eesti kõrgkoolide kursuste nõustaja. "
            "Kui küsimus pole asjakohane ütle et küsiksid ainult kursuste kohta. "
            "Vasta nagu YODA"
            f"{filter_text}\n\n"
            f"{context_text}\n\n"
            "Vasta eesti keeles. Too iga kursuse kohta välja: nimi, kood, maht (EAP), keel ja asukoht. "
            "Kuva kursuse link markdown formaadis: [AINE_KOOD](https://ois2.ut.ee/#/courses/AINE_KOOD) "
            "kus AINE_KOOD on täpne kood tabelist (nt MTAT.03.206). "
            "Kui sobivaid kursusi ei leitud, soovita filtrite laiendamist."
        ),
    }

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    messages_to_send = [system_msg] + messages

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages_to_send,
        temperature=0.6,
        max_tokens=1800,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            yield delta

        if hasattr(chunk, "usage") and chunk.usage:
            yield {
                "usage": {
                    "input": chunk.usage.prompt_tokens,
                    "output": chunk.usage.completion_tokens,
                }
            }
