"""
=============================================================
  VetDesk AI — Phase 3 Agent
  vet_agent.py
=============================================================
  This is the complete agent. It combines:
    - Phase 2: vet_classifier.py  → triage classification
    - Phase 3: vet_sop_knowledge.py → SOP knowledge base
              + TF-IDF RAG engine   → question answering

  What it does with each message:
    1. Checks if it is a booking request  → collects details
    2. Runs the triage classifier         → flags emergencies
    3. Searches the SOP with TF-IDF       → answers questions
    4. Combines everything into one reply

  Zero external dependencies beyond scikit-learn.
  No API key. No internet required.

  To run:
    pip install scikit-learn
    python vet_agent.py
=============================================================
"""

import re
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import Phase 2 classifier
from vet_classifier import classify as triage_classify

# Import Phase 3 knowledge base - Simplified to one line to avoid bracket errors
from vet_sop_knowledge import CHUNKS, CLINIC_NAME, CLINIC_PHONE, EMERGENCY_LINE, WHATSAPP, ADDRESS

# ─────────────────────────────────────────────────────────────
#  HOW TF-IDF WORKS (plain English)
#  ─────────────────────────────────────────────────────────────
#  TF-IDF = Term Frequency × Inverse Document Frequency
#
#  Term Frequency: how often a word appears in one chunk.
#  Inverse Document Frequency: how rare that word is across
#    ALL chunks. Rare words carry more meaning.
#
#  Example: "rabies" appears in 3 chunks out of 80.
#           "the" appears in all 80 chunks.
#           So "rabies" gets a high IDF score, "the" gets near zero.
#
#  When you ask "how much is the rabies vaccine?":
#    → "rabies" is rare and specific → high score
#    → "vaccine" is moderately rare → medium score
#    → "how", "much", "is", "the" → near-zero scores
#
#  The engine ranks every chunk by how well it matches the
#  question using cosine similarity (angle between vectors).
#  The closest chunk wins. That chunk becomes the answer.
# ─────────────────────────────────────────────────────────────


class VetRAG:
    """
    TF-IDF retrieval engine over the SOP knowledge chunks.
    Build once, query many times.
    """

    def __init__(self, chunks):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # single words AND two-word pairs
            stop_words="english", # ignore "the", "is", "a", etc.
            min_df=1,
        )
        # Build the matrix: rows = chunks, columns = vocabulary
        self.matrix = self.vectorizer.fit_transform(chunks)

    def search(self, query, top_k=2, threshold=0.08):
        """
        Find the most relevant chunks for a query.

        Args:
            query:     The owner's question as a string.
            top_k:     How many chunks to return.
            threshold: Minimum similarity score to count as a match.
                       0.0 = return anything. 1.0 = exact match only.
                       0.08 is a good balance for clinic questions.

        Returns:
            List of (score, chunk_text) tuples, best match first.
            Empty list if nothing passes the threshold.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        # Get top_k indices sorted by score descending
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = scores[idx]
            if score >= threshold:
                results.append((round(float(score), 3), self.chunks[idx]))

        return results


# ─────────────────────────────────────────────────────────────
#  INTENT DETECTION
#  Figures out WHAT the owner is trying to do before deciding
#  how to respond. Keeps the logic clean and explicit.
# ─────────────────────────────────────────────────────────────

BOOKING_PATTERNS = [
    r"(book|schedule|make|arrange|set up).{0,20}(appointment|visit|consult)",
    r"(want|need|like).{0,15}(appointment|book|come in)",
    r"(can i|how do i|where do i).{0,15}(book|schedule)",
    r"(next|first|earliest).{0,10}(available|slot|appointment|opening)",
]

INFO_PATTERNS = [
    r"(how much|what does|cost|price|fee|charge|expensive)",
    r"(what time|when|hours|open|close|opening|closing)",
    r"(do you|can you|do you treat|do you accept|do you see)",
    r"(what is|what are|how do|how long|how often)",
    r"(tell me|explain|describe|what should|need to know)",
]

BOOKING_FIELDS = ["pet_name", "species", "owner_name", "phone", "reason", "date", "time"]
BOOKING_QUESTIONS = {
    "pet_name":   "What is your pet's name?",
    "species":    "What type of animal is it? (dog, cat, rabbit, or bird)",
    "owner_name": "What is your name?",
    "phone":      "What is the best phone number to reach you?",
    "reason":     "What is the reason for the visit?",
    "date":       "What date would you prefer? (e.g. Monday, April 14)",
    "time":       "What time works best? Morning (9–11am) or afternoon (1–4pm)?",
}


def detect_intent(message):
    """Returns: 'booking', 'info', 'triage', or 'unclear'"""
    norm = message.lower()
    for pat in BOOKING_PATTERNS:
        if re.search(pat, norm):
            return "booking"
    for pat in INFO_PATTERNS:
        if re.search(pat, norm):
            return "info"
    return "unclear"


# ─────────────────────────────────────────────────────────────
#  BOOKING STATE MACHINE
#  Tracks partial bookings across multiple messages.
#  In a real deployment this would be per-user session.
# ─────────────────────────────────────────────────────────────

class BookingSession:
    def __init__(self):
        self.active = False
        self.data = {f: None for f in BOOKING_FIELDS}
        self.current_field = None

    def start(self):
        self.active = True
        self.current_field = "pet_name"
        return (
            f"I can help you book an appointment at {CLINIC_NAME}!\n\n"
            + BOOKING_QUESTIONS["pet_name"]
        )

    def next_field(self):
        """Returns the next unfilled field, or None if complete."""
        for field in BOOKING_FIELDS:
            if self.data[field] is None:
                return field
        return None

    def receive(self, message):
        """Store the answer to the current question, ask the next."""
        if self.current_field:
            self.data[self.current_field] = message.strip()

        next_f = self.next_field()
        if next_f:
            self.current_field = next_f
            return BOOKING_QUESTIONS[next_f]
        else:
            return self._confirm()

    def _confirm(self):
        self.active = False
        d = self.data
        summary = (
            f"All set! Here is your booking summary:\n\n"
            f"  Pet:     {d['pet_name']} ({d['species']})\n"
            f"  Owner:   {d['owner_name']}\n"
            f"  Phone:   {d['phone']}\n"
            f"  Reason:  {d['reason']}\n"
            f"  Date:    {d['date']}\n"
            f"  Time:    {d['time']}\n\n"
            f"Please arrive 10 minutes early with any previous vaccine records.\n"
            f"If anything changes, call us at {CLINIC_PHONE}.\n"
            f"See you soon!"
        )
        return summary

    def reset(self):
        self.__init__()


# ─────────────────────────────────────────────────────────────
#  MAIN AGENT
#  Combines triage + RAG + booking into a single respond() call.
# ─────────────────────────────────────────────────────────────

class VetAgent:
    """
    The complete VetDesk AI agent.
    Instantiate once, call respond(message) for each owner message.
    """

    def __init__(self):
        print("Building knowledge index...", end=" ", flush=True)
        self.rag = VetRAG(CHUNKS)
        print("ready.")
        self.booking = BookingSession()

    def respond(self, message):
        """
        Process one owner message and return the agent's reply.

        Decision flow:
          1. If a booking is in progress → continue collecting fields
          2. Run triage classifier → if EMERGENCY → stop, urgent message
          3. Detect intent
             → booking intent   → start booking flow
             → info intent      → search SOP with RAG
             → unclear          → search SOP anyway, offer booking
        """

        # ── 1. Booking in progress ───────────────────────────
        if self.booking.active:
            return self.booking.receive(message)

        # ── 2. Triage check ──────────────────────────────────
        triage = triage_classify(message)
        level  = triage["classification"]

        if level == "EMERGENCY":
            return (
                f"EMERGENCY — Please act immediately.\n\n"
                f"Call our 24/7 emergency line now: {EMERGENCY_LINE}\n"
                f"Or bring your pet to {ADDRESS} RIGHT NOW.\n\n"
                f"Do not wait. Every minute matters."
            )

        # ── 3. Intent detection ──────────────────────────────
        intent = detect_intent(message)

        # ── 3a. Booking intent ───────────────────────────────
        if intent == "booking":
            return self.booking.start()

        # ── 3b. Info intent or unclear → search SOP ─────────
        results = self.rag.search(message, top_k=2, threshold=0.08)

        if results:
            # Build reply from top matching chunks
            reply_parts = []
            for score, chunk in results:
                reply_parts.append(chunk)

            reply = "\n\n".join(reply_parts)

            # Append a MODERATE flag if relevant
            if level == "MODERATE":
                reply += (
                    f"\n\nNote: Based on what you've described, your pet "
                    f"should be seen by a vet today or tomorrow. "
                    f"Please call {CLINIC_PHONE} to book the earliest slot."
                )

            # Offer to book if they haven't already
            if intent != "booking":
                reply += f"\n\nWould you like to book an appointment? Just say 'book appointment' or call {CLINIC_PHONE}."

            return reply

        # ── 3c. No match found ───────────────────────────────
        fallback = (
            f"I'm not sure I have the right information for that.\n"
            f"Please call us directly at {CLINIC_PHONE} and a team "
            f"member will be happy to help."
        )

        if level == "MODERATE":
            fallback += (
                f"\n\nBased on what you've described, your pet should "
                f"be seen by a vet today or tomorrow."
            )

        return fallback


# ─────────────────────────────────────────────────────────────
#  TERMINAL COLOURS
# ─────────────────────────────────────────────────────────────

C = {
    "agent":  "\033[96m",   # cyan
    "owner":  "\033[97m",   # white
    "system": "\033[90m",   # grey
    "R":      "\033[0m",
    "B":      "\033[1m",
}

def print_exchange(role, text):
    if role == "agent":
        label = f"{C['agent']}{C['B']}VetDesk AI :{C['R']} "
    else:
        label = f"{C['owner']}You        :{C['R']} "
    # Indent continuation lines
    indented = text.replace("\n", "\n             ")
    print(f"\n{label}{indented}")


# ─────────────────────────────────────────────────────────────
#  TEST SUITE — covers all three layers working together
# ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # Emergency — classifier fires, RAG never runs
    ("my cat is gasping and her gums are blue",
     "EMERGENCY"),

    # Pricing questions — RAG should answer from chunks
    ("how much does it cost to spay a cat",
     "spay"),
    ("what is the price of a rabies vaccine",
     "rabies"),
    ("how much is microchipping",
     "microchip"),

    # Hours / logistics
    ("what time do you open on saturday",
     "9:00 AM"),
    ("are you open on sundays",
     "closed"),

    # Animals / services
    ("do you treat rabbits",
     "rabbit"),
    ("do you see birds",
     "bird"),

    # Clinical knowledge from SOP
    ("how long should i fast my dog before surgery",
     "8 to 12 hours"),
    ("can i fast my rabbit before surgery",
     "NOT"),   # should say do NOT fast rabbits

    # Triage + advice
    ("my rabbit hasnt eaten anything today",
     "GI stasis"),

    # Booking intent
    ("id like to book an appointment",
     "book"),
]


def run_tests(agent):
    print(f"\n{'=' * 65}")
    print("  VetDesk AI — Phase 3 Agent Test Suite")
    print(f"{'=' * 65}")

    passed = failed = 0
    failures = []

    for message, expected_fragment in TEST_CASES:
        reply = agent.respond(message)
        agent.booking.reset()   # reset booking after each test
        ok = expected_fragment.lower() in reply.lower()
        sym = "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"
        print(f"  {sym}  {message[:55]}")
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((message, expected_fragment, reply[:80]))

    print(f"\n  Results: {passed}/{len(TEST_CASES)} passed")
    if failures:
        print("\n  Failed cases:")
        for msg, exp, got in failures:
            print(f"    Query   : '{msg}'")
            print(f"    Expected: '{exp}' in reply")
            print(f"    Got     : '{got}...'")
    print(f"{'=' * 65}\n")
    return passed, failed


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
# ─────────────────────────────────────────────────────────────

def interactive_mode(agent):
    print(f"\n{'=' * 65}")
    print(f"  VetDesk AI — Full Agent  (Phase 2 + Phase 3)")
    print(f"  Powered by: TF-IDF RAG + keyword/regex triage")
    print(f"  Type 'quit' to exit | 'reset' to clear booking state")
    print(f"{'=' * 65}")

    while True:
        try:
            msg = input(f"\n{C['owner']}You: {C['R']}").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not msg:
            continue
        if msg.lower() in ("quit", "exit", "q"):
            break
        if msg.lower() == "reset":
            agent.booking.reset()
            print(f"{C['system']}Booking state cleared.{C['R']}")
            continue

        reply = agent.respond(msg)
        print_exchange("agent", reply)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = VetAgent()

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    if mode == "test":
        run_tests(agent)
    elif mode == "interactive":
        interactive_mode(agent)
    else:
        run_tests(agent)
        interactive_mode(agent)
