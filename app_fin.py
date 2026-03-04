import streamlit as st

from backend import (
    build_mask,
    filter_summary_text,
    format_context,
    load_resources,
    run_llm_stream,
    semantic_search,
)
from history import save_session
from ui import refresh_usage_totals, render_chat_history, render_sidebar

# ── Page config
st.set_page_config(page_title="AI Kursuse Nõustaja", page_icon="🎓")
st.title("🎓 AI Kursuse Nõustaja")
st.caption("RAG süsteem koos metaandmete filtreerimisega")

# ── Load data & model (cached)
embedder, df, embeddings_df = load_resources()

# ── Sidebar (returns api key + active filters)
api_key, filters = render_sidebar(df)

# ── Render existing chat messages
render_chat_history()

# ── Handle new user input
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "token_input": 0, "token_output": 0}
    )
    refresh_usage_totals()
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            msg = "Palun sisesta OpenRouter API võti külgribal."
            st.error(msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": msg,
                    "token_input": 0,
                    "token_output": 0,
                }
            )
            refresh_usage_totals()
            save_session(st.session_state.session_id, st.session_state.messages)
            st.stop()

        with st.spinner("Otsin sobivaid kursusi..."):
            # 1. Filter
            mask = build_mask(
                df,
                filters["eristav"],
                filters["semester"],
                filters["aste"],
                filters["keel"],
                filters["asukoht"],
            )
            filtered_df = df[mask]

            # 2. Semantic search
            if filtered_df.empty:
                context_text = "Sobivaid kursusi ei leitud – filtrid on liiga kitsad."
            else:
                top = semantic_search(filtered_df, embeddings_df, embedder, prompt)
                context_text = (
                    format_context(top)
                    if not top.empty
                    else "Embeddings puuduvad leitud kursustel."
                )

            # 3. Build filter summary
            filter_text = filter_summary_text(
                filters["eristav"],
                filters["semester"],
                filters["aste"],
                filters["keel"],
                filters["asukoht"],
            )

        # 4. Stream LLM response
        full_response = ""
        placeholder = st.empty()
        turn_input_tokens = 0
        turn_output_tokens = 0

        try:
            for chunk in run_llm_stream(
                api_key, st.session_state.messages, filter_text, context_text
            ):
                if isinstance(chunk, str):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                elif isinstance(chunk, dict) and "usage" in chunk:
                    turn_input_tokens = int(chunk["usage"]["input"] or 0)
                    turn_output_tokens = int(chunk["usage"]["output"] or 0)

            placeholder.markdown(full_response)

            # Attribute request prompt tokens to latest user message and
            # completion tokens to assistant response message.
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages[-1]["token_input"] = turn_input_tokens
                st.session_state.messages[-1]["token_output"] = 0

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "token_input": 0,
                    "token_output": turn_output_tokens,
                }
            )
            refresh_usage_totals()

            # 5. Persist to history
            save_session(st.session_state.session_id, st.session_state.messages)

        except Exception as e:
            err = f"Viga LLM-iga ühenduses: {e}"
            st.error(err)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": err,
                    "token_input": 0,
                    "token_output": 0,
                }
            )
            refresh_usage_totals()
            save_session(st.session_state.session_id, st.session_state.messages)
