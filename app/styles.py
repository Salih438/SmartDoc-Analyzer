"""Küçük, uygulama genelinde kullanılan görsel iyileştirmeler."""

from __future__ import annotations

import streamlit as st


_CSS = """
<style>
.sd-card { border: 1px solid rgba(99, 102, 241, .16); border-radius: 16px;
  padding: 1.1rem 1.25rem; background: rgba(255,255,255,.72);
  box-shadow: 0 5px 18px rgba(31,41,55,.05); }
.sd-card h3 { margin: .1rem 0 .45rem; font-size: 1.08rem; }
.sd-title { font-size: 1.85rem; line-height: 1.15; font-weight: 700; letter-spacing: -.035em; }
.sd-subtitle { color: #64748b; margin: .35rem 0 1.5rem; }
.sd-status { display: inline-block; padding: .35rem .75rem; border-radius: 999px;
  font-weight: 600; font-size: .88rem; }
.sd-ok { color: #166534; background: #dcfce7; }
.sd-warn { color: #92400e; background: #fef3c7; }
.sd-error { color: #991b1b; background: #fee2e2; }
div.stButton > button, div.stDownloadButton > button { border-radius: 10px; }
</style>
"""


def inject_styles() -> None:
    """Enjekte edilen tek CSS bloğunu uygula."""
    st.markdown(_CSS, unsafe_allow_html=True)
