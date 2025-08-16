set -euo pipefail

FILE="app/dashboard.py"

perl -0777 -i -pe 's/\n?st\.markdown\(\s*?\n\s*"""\s*?\n\s*<hr>.*?All rights reserved\.\s*?\n\s*""",\s*?\n\s*unsafe_allow_html=True\s*?\n\)\s*?\n//gs' "$FILE"

sed -i '' '/# BEGIN-AR-COPYRIGHT/,/# END-AR-COPYRIGHT/d' "$FILE" 2>/dev/null || true

cat <<'PY' >> "$FILE"
# BEGIN-AR-COPYRIGHT
import streamlit as st
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:18px; color:gray;">
    © 2025 Aditya Ravi — All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
PY

echo "✅ Ensured single copyright footer in $FILE"
