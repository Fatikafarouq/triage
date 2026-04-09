"""
=============================================================
  VetDesk AI — Streamlit Web App  (Phase 4)
  app.py
=============================================================
  Wraps the Phase 3 agent in a clean chat UI.

  To run locally:
    pip install streamlit scikit-learn
    streamlit run app.py

  To deploy free on Streamlit Cloud:
    1. Push all 4 files to a GitHub repo:
         app.py
         vet_agent.py
         vet_classifier.py
         vet_sop_knowledge.py
    2. Go to share.streamlit.io
    3. Connect your repo → deploy
    4. You get a public URL instantly
=============================================================
"""

import streamlit as st
from vet_agent import VetAgent
from vet_sop_knowledge import CLINIC_NAME, CLINIC_PHONE, EMERGENCY_LINE

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=f"VetDesk AI — {CLINIC_NAME}",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS
#  Mobile-first, clean card design. Triage badges are coloured.
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Global font */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }

  /* Remove default streamlit top padding */
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

  /* Header bar */
  .vet-header {
    background: #0f4c75;
    border-radius: 14px;
    padding: 18px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .vet-header-text h2 {
    color: #ffffff;
    margin: 0;
    font-size: 1.3rem;
    font-weight: 700;
  }
  .vet-header-text p {
    color: #a8d1f0;
    margin: 2px 0 0;
    font-size: 0.82rem;
  }
  .vet-logo {
    font-size: 2.2rem;
    line-height: 1;
  }

  /* Chat bubbles */
  .msg-row { display: flex; margin-bottom: 12px; }
  .msg-row.user  { justify-content: flex-end; }
  .msg-row.agent { justify-content: flex-start; }

  .bubble {
    max-width: 82%;
    padding: 11px 16px;
    border-radius: 18px;
    font-size: 0.9rem;
    line-height: 1.55;
    white-space: pre-wrap;
    word-wrap: break-word;
  }
  .bubble.user {
    background: #0f4c75;
    color: #ffffff;
    border-bottom-right-radius: 4px;
  }
  .bubble.agent {
    background: #f0f4f8;
    color: #1a1a2e;
    border-bottom-left-radius: 4px;
    border: 1px solid #e0e8f0;
  }

  /* Triage badges inside agent bubbles */
  .badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-bottom: 6px;
    letter-spacing: 0.04em;
  }
  .badge.emergency { background: #fee2e2; color: #991b1b; }
  .badge.moderate  { background: #fef3c7; color: #92400e; }
  .badge.routine   { background: #d1fae5; color: #065f46; }
  .badge.info      { background: #dbeafe; color: #1e40af; }
  .badge.booking   { background: #ede9fe; color: #4c1d95; }

  /* Quick reply chips */
  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 10px;
    margin-bottom: 4px;
  }

  /* Emergency banner */
  .emergency-banner {
    background: #fee2e2;
    border: 1.5px solid #f87171;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 16px;
  }
  .emergency-banner strong { color: #991b1b; font-size: 1rem; }
  .emergency-banner p { color: #7f1d1d; margin: 6px 0 0; font-size: 0.88rem; }

  /* Sidebar info cards */
  .info-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 0.84rem;
    color: #334155;
  }
  .info-card strong { color: #0f4c75; }

  /* Input area override */
  .stTextInput > div > div > input {
    border-radius: 24px !important;
    border: 1.5px solid #cbd5e1 !important;
    padding: 10px 18px !important;
  }

  /* Hide streamlit branding */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SESSION STATE INIT
#  Streamlit reruns the whole script on every interaction.
#  st.session_state persists data across reruns.
# ─────────────────────────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = VetAgent()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "agent",
            "text": (
                f"Hello! I'm VetDesk AI, the virtual receptionist for "
                f"*{CLINIC_NAME}*.\n\n"
                f"I can help you with:\n"
                f"• 🚨 Pet emergencies — I'll flag urgent cases immediately\n"
                f"• 📅 Booking appointments\n"
                f"• 💬 Questions about our services, prices, and hours\n\n"
                f"How can I help you today?"
            ),
            "badge": "info",
        }
    ]

if "last_triage" not in st.session_state:
    st.session_state.last_triage = None


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────

st.markdown(f"""
<div class="vet-header">
  <div class="vet-logo">🐾</div>
  <div class="vet-header-text">
    <h2>VetDesk AI</h2>
    <p>{CLINIC_NAME} · Virtual Receptionist</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR — clinic quick info
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📋 Clinic Info")
    st.markdown(f"""
    <div class="info-card">
      <strong>🏥 {CLINIC_NAME}</strong><br>
      📞 {CLINIC_PHONE}
    </div>
    <div class="info-card">
      <strong>🚨 Emergency (24/7)</strong><br>
      {EMERGENCY_LINE}
    </div>
    <div class="info-card">
      <strong>🕐 Hours</strong><br>
      Mon–Fri: 8am – 6pm<br>
      Saturday: 9am – 4pm<br>
      Sunday: Closed
    </div>
    <div class="info-card">
      <strong>🐾 We treat</strong><br>
      Dogs · Cats · Rabbits · Birds
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    if st.button("🔄 Start new conversation"):
        st.session_state.messages = [st.session_state.messages[0]]
        st.session_state.agent = VetAgent()
        st.session_state.last_triage = None
        st.rerun()


# ─────────────────────────────────────────────────────────────
#  EMERGENCY BANNER
#  Shown persistently at top if last triage was EMERGENCY
# ─────────────────────────────────────────────────────────────

if st.session_state.last_triage == "EMERGENCY":
    st.markdown(f"""
    <div class="emergency-banner">
      <strong>🚨 Emergency detected</strong>
      <p>Call our 24/7 emergency line immediately: <strong>{EMERGENCY_LINE}</strong></p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  CHAT HISTORY
# ─────────────────────────────────────────────────────────────

BADGE_LABELS = {
    "emergency": "🚨 EMERGENCY",
    "moderate":  "⚠️ MODERATE",
    "routine":   "✅ ROUTINE",
    "info":      "💬 Info",
    "booking":   "📅 Booking",
}

chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        role  = msg["role"]
        text  = msg["text"]
        badge = msg.get("badge")

        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
              <div class="bubble user">{text}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            badge_html = ""
            if badge and badge in BADGE_LABELS:
                badge_html = f'<div class="badge {badge}">{BADGE_LABELS[badge]}</div>'
            st.markdown(f"""
            <div class="msg-row agent">
              <div class="bubble agent">{badge_html}{text}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  QUICK REPLY CHIPS
#  Shown only when no booking is in progress
# ─────────────────────────────────────────────────────────────

QUICK_REPLIES = [
    "Book an appointment",
    "What are your hours?",
    "How much is a rabies vaccine?",
    "Do you treat rabbits?",
    "My dog ate chocolate",
]

if not st.session_state.agent.booking.active:
    st.markdown('<div class="chip-row">', unsafe_allow_html=True)
    cols = st.columns(len(QUICK_REPLIES))
    for i, qr in enumerate(QUICK_REPLIES):
        with cols[i]:
            if st.button(qr, key=f"chip_{qr}", use_container_width=True):
                st.session_state._pending_input = qr
    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  MESSAGE INPUT
# ─────────────────────────────────────────────────────────────

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            label="message",
            label_visibility="collapsed",
            placeholder="Type your message here…",
        )
    with col2:
        submitted = st.form_submit_button("Send", use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PROCESS MESSAGE
# ─────────────────────────────────────────────────────────────

def determine_badge(reply: str, triage_level: str) -> str:
    """Pick the right badge colour for an agent reply."""
    if triage_level == "EMERGENCY":
        return "emergency"
    if triage_level == "MODERATE":
        return "moderate"
    booking_words = ["What is your pet", "What type of animal", "What is your name",
                     "What is the best phone", "What is the reason", "What date",
                     "What time", "booking summary", "See you soon"]
    if any(w in reply for w in booking_words):
        return "booking"
    if triage_level == "ROUTINE":
        return "routine"
    return "info"


def handle_message(text: str):
    if not text.strip():
        return

    # Add user message to history
    st.session_state.messages.append({"role": "user", "text": text})

    # Get triage level for badge (run classifier separately — agent also runs it)
    from vet_classifier import classify as triage_classify
    triage = triage_classify(text)
    level  = triage["classification"]
    st.session_state.last_triage = level

    # Get agent reply
    reply = st.session_state.agent.respond(text)
    badge = determine_badge(reply, level)

    st.session_state.messages.append({
        "role":  "agent",
        "text":  reply,
        "badge": badge,
    })

    st.rerun()


# Handle quick reply chips
if hasattr(st.session_state, "_pending_input") and st.session_state._pending_input:
    pending = st.session_state._pending_input
    st.session_state._pending_input = None
    handle_message(pending)

# Handle typed input
if submitted and user_input:
    handle_message(user_input)
