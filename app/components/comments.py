# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
from __future__ import annotations
import requests
import streamlit as st
from typing import Optional

def render_comments(
    *,
    thread_id: str,
    author_default: str,
    api_base: str,
    use_api: bool,
    key_prefix: str = "comments",
) -> None:
    """
    Simple comments panel with a thread id.
    - Uses unique keys via key_prefix so you can render it multiple times.
    - Never mutates session_state for the input after widget creation.
    """
    list_key  = f"{key_prefix}_list"
    input_key = f"{key_prefix}_input"
    send_key  = f"{key_prefix}_send"

    if not use_api:
        st.info("API mode is off. Set OPTIM_API_URL to enable comments.")
        return

    # --- Load comments ---
    try:
        r = requests.get(f"{api_base}/comments/{thread_id}", timeout=5)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as e:
        st.error(f"Load failed: {e}")
        items = []

    # --- Render list ---
    if not items:
        st.caption("No comments yet. Be the first to comment!")
    else:
        for it in items:
            with st.chat_message(name=it.get("author") or "Anonymous"):
                st.write(it.get("text", ""))
                st.caption(str(it.get("created_at", "")))

    st.divider()

    # Initialize the input *before* creating the widget (prevents duplicate-key error)
    if input_key not in st.session_state:
        st.session_state[input_key] = ""

    # Use chat_input (nice UX). It has its own submit action.
    msg = st.chat_input("Write a comment…", key=input_key)

    # If you prefer a text_input + button pattern, uncomment below and comment the chat_input above:
    # msg = st.text_input("Write a comment…", key=input_key)
    # if st.button("Post", key=send_key):
    #     _submit = st.session_state.get(input_key, "")
    #     if _submit:
    #         _post_comment(api_base, thread_id, author_default, _submit)
    #         st.session_state.pop(input_key, None)
    #         st.rerun()

    # chat_input returns the text only on submit. Post & clear safely.
    if msg:
        _ok, _err = _post_comment(api_base, thread_id, author_default, msg)
        if _ok:
            # Clear by removing the key, then rerun (so the widget re-mounts fresh).
            st.session_state.pop(input_key, None)
            st.rerun()
        else:
            st.error(_err)


def _post_comment(api_base: str, thread_id: str, author: str, text: str) -> tuple[bool, Optional[str]]:
    try:
        payload = {"strategy": thread_id, "author": author or "Anonymous", "text": text}
        r = requests.post(f"{api_base}/comments", json=payload, timeout=5)
        r.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)