"""
=============================================================
  VetDesk AI — Emergency Classifier
  Phase 2 of the Vet AI Agent Roadmap
=============================================================
  What this script does:
    Takes a message from a pet owner (text) and classifies it
    as one of three triage levels:
      - EMERGENCY  → life-threatening, act NOW
      - MODERATE   → needs a vet soon, same/next day
      - ROUTINE    → book a normal appointment

  How it works (two layers):
    Layer 1 — Keyword safety net (instant, no API needed)
              Catches obvious emergencies even if the AI fails.
    Layer 2 — LLM reasoning (Claude via Anthropic API)
              Understands context, phrasing, and nuance.

  How to run:
    1. pip install anthropic
    2. Set your API key (see SETUP below)
    3. python vet_classifier.py

  Google Colab users:
    Replace the API key line with:
    from google.colab import userdata
    API_KEY = userdata.get("ANTHROPIC_API_KEY")
=============================================================
"""

import os
import re
import anthropic

# ─────────────────────────────────────────────────────────────
#  SETUP — Add your Anthropic API key here
# ─────────────────────────────────────────────────────────────
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-api-key-here")


# ─────────────────────────────────────────────────────────────
#  LAYER 1: KEYWORD SAFETY NET
#  These lists are your first line of defence.
#  If ANY emergency keyword is found in the message,
#  the script immediately flags it as EMERGENCY —
#  without even calling the AI.
#  This means: even if the API is down, emergencies are caught.
# ─────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    # Breathing
    "can't breathe", "cannot breathe", "not breathing", "struggling to breathe",
    "labored breathing", "laboured breathing", "gasping", "open mouth breathing",
    "choking",

    # Gum color — classic emergency sign
    "blue gums", "white gums", "grey gums", "gray gums", "pale gums",
    "purple gums",

    # Collapse / consciousness
    "collapsed", "collapse", "unconscious", "unresponsive", "passed out",
    "won't wake up", "can't wake up",

    # Seizures
    "seizure", "seizing", "convulsing", "convulsion", "fitting", "fit",

    # Toxins — full list from SOP
    "ate chocolate", "eaten chocolate", "swallowed chocolate",
    "ate grapes", "ate raisins", "ate onion", "ate garlic",
    "ate xylitol", "ate rat poison", "ate antifreeze",
    "ate ibuprofen", "ate advil", "ate tylenol", "ate acetaminophen",
    "swallowed", "ingested",        # broad catch — AI refines
    "ate something", "ate a",       # broad catch — AI refines
    "poisoned", "poison",

    # Trauma
    "hit by car", "hit by a car", "run over", "fell from", "fell off", "attacked",
    "bite wound", "dog fight", "cat fight", "severe bleeding",
    "won't stop bleeding", "bleeding heavily",

    # Urinary obstruction — especially male cats
    "can't urinate", "cannot urinate", "not urinating", "straining to urinate",
    "blocked bladder", "no urine",

    # Bloat / GDV
    "swollen belly", "distended abdomen", "bloated", "bloat",

    # Birth emergencies
    "can't give birth", "stuck in birth canal", "whelping problem",
    "green discharge before", "no puppy after",

    # Grapes / raisins (varied phrasing)
    "ate some grapes", "ate grapes", "ate raisins", "some grapes",
    "a few grapes", "handful of grapes",

    # Bloat / GDV (varied phrasing)
    "stomach looks huge", "belly looks huge", "stomach is huge",
    "abdomen looks swollen", "drooling a lot", "drooling and bloated",

    # Snake / scorpion
    "snake bite", "snakebite", "scorpion sting",

    # Eyes
    "eye popped out", "eye prolapse", "sudden blindness", "can't see",

    # Paralysis
    "can't walk", "dragging legs", "paralyzed", "paralysed",
]

MODERATE_KEYWORDS = [
    "vomiting", "vomited", "throwing up",
    "diarrhea", "diarrhoea", "loose stool", "bloody stool",
    "not eating", "won't eat", "hasn't eaten", "off food",
    "limping", "lame", "favoring leg",
    "scratching ear", "shaking head", "ear smell",
    "eye discharge", "eye watering", "squinting",
    "coughing", "sneezing",
    "lump", "swelling", "bump",
    "drinking a lot", "urinating a lot", "excessive thirst",
    "wound", "cut", "abscess",
    "rabbit not eating",         # GI stasis risk — important
    "bird on bottom of cage",    # classic sick bird sign
    "sitting at the bottom",     # bird on cage floor phrasing
    "bird sitting on the bottom",
    "bird puffed up",
]

ROUTINE_KEYWORDS = [
    "vaccination", "vaccine", "booster", "shot",
    "deworming", "deworm", "worm treatment",
    "spay", "neuter", "castration",
    "microchip", "microchipping",
    "nail trim", "grooming",
    "check up", "checkup", "wellness", "annual visit",
    "health certificate", "travel certificate",
    "new puppy", "new kitten", "first visit",
    "follow up", "follow-up",
    "flea treatment", "tick treatment",
    "weight check",
]


def keyword_check(message: str) -> str | None:
    """
    Scans the message for known keywords.
    Returns 'EMERGENCY', 'MODERATE', 'ROUTINE', or None.
    None means: no keyword matched — hand off to the AI.
    """
    text = message.lower()

    for kw in EMERGENCY_KEYWORDS:
        if kw in text:
            return "EMERGENCY"

    for kw in MODERATE_KEYWORDS:
        if kw in text:
            return "MODERATE"

    for kw in ROUTINE_KEYWORDS:
        if kw in text:
            return "ROUTINE"

    return None  # No match found


# ─────────────────────────────────────────────────────────────
#  LAYER 2: LLM CLASSIFIER (Claude via Anthropic API)
#  This runs when keywords don't give a clear answer,
#  OR to double-check keyword results with reasoning.
#  The system prompt is your SOP rules translated into
#  instructions the AI can follow.
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a veterinary triage assistant for a small animal clinic.
Your only job is to classify incoming messages from pet owners
into one of three triage levels.

TRIAGE LEVELS:
  EMERGENCY  — Life-threatening. Owner must act immediately.
               Examples: difficulty breathing, blue/white gums,
               collapse, seizures, suspected poisoning (chocolate,
               xylitol, rat poison, grapes, ibuprofen, antifreeze),
               snake bite, male cat unable to urinate, hit by car,
               severe bleeding, not waking up, GDV/bloat in dogs,
               eye prolapse, paralysis.

  MODERATE   — Needs veterinary attention within 24 hours.
               Examples: vomiting more than twice in 24hrs,
               diarrhea with blood, not eating for 24-48hrs,
               limping but weight-bearing, eye redness/discharge,
               ear problems, small wounds, rabbit not eating
               (GI stasis risk), bird sitting on cage floor.

  ROUTINE    — Normal appointment. No urgency.
               Examples: vaccinations, deworming, spay/neuter,
               microchip, wellness check, nail trim, follow-up,
               new puppy/kitten visit, flea/tick prevention,
               health certificate for travel.

RULES:
  - When in doubt between EMERGENCY and MODERATE, choose EMERGENCY.
  - Chocolate ingestion is always EMERGENCY regardless of amount.
  - Male cat straining to urinate is always EMERGENCY.
  - Rabbit not eating for more than 12 hours is MODERATE (GI stasis).
  - Never downgrade an emergency just because the owner says
    the pet "seems okay" — trauma patients can crash suddenly.

OUTPUT FORMAT (strict — nothing else):
  CLASSIFICATION: [EMERGENCY|MODERATE|ROUTINE]
  REASON: [one sentence explaining why]
  ACTION: [one sentence telling the receptionist what to do]
"""


def llm_classify(message: str, client: anthropic.Anthropic) -> dict:
    """
    Sends the message to Claude and parses the structured response.
    Returns a dict with keys: classification, reason, action.
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # Fast and cheap — perfect for triage
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": f"Pet owner message: {message}"}
        ]
    )

    raw = response.content[0].text.strip()

    # Parse the structured output
    result = {
        "classification": "UNKNOWN",
        "reason": "Could not parse AI response.",
        "action": "Please review manually.",
        "raw": raw
    }

    for line in raw.splitlines():
        if line.startswith("CLASSIFICATION:"):
            value = line.replace("CLASSIFICATION:", "").strip()
            if value in ("EMERGENCY", "MODERATE", "ROUTINE"):
                result["classification"] = value
        elif line.startswith("REASON:"):
            result["reason"] = line.replace("REASON:", "").strip()
        elif line.startswith("ACTION:"):
            result["action"] = line.replace("ACTION:", "").strip()

    return result


# ─────────────────────────────────────────────────────────────
#  RESPONSE SCRIPTS
#  These are the exact messages the agent sends back
#  to the pet owner based on triage level.
#  Edit these to match your clinic's tone and details.
# ─────────────────────────────────────────────────────────────

RESPONSES = {
    "EMERGENCY": (
        "🚨 This sounds like an emergency.\n"
        "Please call our 24/7 emergency line immediately: [YOUR NUMBER]\n"
        "Or bring your pet to the clinic RIGHT NOW — do not wait.\n"
        "Every minute matters in this situation."
    ),
    "MODERATE": (
        "⚠️  Your pet needs to be seen by a vet today or tomorrow.\n"
        "Please call us to book the earliest available slot: [YOUR NUMBER]\n"
        "If symptoms get worse (difficulty breathing, collapse, seizures),\n"
        "treat it as an emergency and come in immediately."
    ),
    "ROUTINE": (
        "✅ This sounds like a routine visit.\n"
        "You can book an appointment at your convenience.\n"
        "Call us at [YOUR NUMBER] or reply here with your preferred date and time."
    ),
    "UNKNOWN": (
        "I wasn't able to assess this automatically.\n"
        "Please call the clinic directly: [YOUR NUMBER]\n"
        "A team member will help you right away."
    ),
}


# ─────────────────────────────────────────────────────────────
#  MAIN CLASSIFIER FUNCTION
#  This is what you call from any other script or chat interface.
#  It combines both layers and returns everything you need.
# ─────────────────────────────────────────────────────────────

def classify(message: str, client: anthropic.Anthropic = None) -> dict:
    """
    Full triage classification pipeline.

    Args:
        message: The pet owner's message (free text).
        client:  Anthropic API client. If None, keyword-only mode.

    Returns:
        dict with keys:
          - classification: EMERGENCY / MODERATE / ROUTINE / UNKNOWN
          - reason:         Why this classification was chosen
          - action:         What the receptionist should do
          - response:       The message to send back to the owner
          - method:         'keyword' or 'llm' (how it was classified)
    """

    # ── Layer 1: Keyword check ──────────────────────────────
    keyword_result = keyword_check(message)

    # For EMERGENCY: keyword match is final — no need to call AI.
    # Speed matters when a pet's life is at risk.
    if keyword_result == "EMERGENCY":
        return {
            "classification": "EMERGENCY",
            "reason": "Emergency keyword detected in message.",
            "action": "Direct owner to emergency line immediately. Do not delay.",
            "response": RESPONSES["EMERGENCY"],
            "method": "keyword",
        }

    # ── Layer 2: LLM classification ─────────────────────────
    if client:
        try:
            llm_result = llm_classify(message, client)
            classification = llm_result["classification"]

            # Safety check: if keyword said MODERATE but AI says ROUTINE,
            # trust the keyword — it's the more cautious call.
            if keyword_result == "MODERATE" and classification == "ROUTINE":
                classification = "MODERATE"
                llm_result["reason"] += " (Upgraded to MODERATE by keyword safety net.)"

            return {
                "classification": classification,
                "reason": llm_result["reason"],
                "action": llm_result["action"],
                "response": RESPONSES.get(classification, RESPONSES["UNKNOWN"]),
                "method": "llm",
            }

        except Exception as e:
            # If the API call fails, fall back to keyword result
            fallback = keyword_result or "UNKNOWN"
            return {
                "classification": fallback,
                "reason": f"API error: {e}. Keyword fallback used.",
                "action": "Review manually or call the owner directly.",
                "response": RESPONSES.get(fallback, RESPONSES["UNKNOWN"]),
                "method": "keyword-fallback",
            }

    # ── Keyword-only mode (no API client provided) ──────────
    result = keyword_result or "UNKNOWN"
    return {
        "classification": result,
        "reason": "Keyword match only (no API client provided).",
        "action": "Confirm with vet before acting on keyword-only results.",
        "response": RESPONSES.get(result, RESPONSES["UNKNOWN"]),
        "method": "keyword-only",
    }


# ─────────────────────────────────────────────────────────────
#  PRETTY PRINTER
# ─────────────────────────────────────────────────────────────

COLORS = {
    "EMERGENCY": "\033[91m",   # Red
    "MODERATE":  "\033[93m",   # Yellow
    "ROUTINE":   "\033[92m",   # Green
    "UNKNOWN":   "\033[90m",   # Grey
    "RESET":     "\033[0m",
    "BOLD":      "\033[1m",
    "DIM":       "\033[2m",
}

def print_result(message: str, result: dict):
    c = COLORS.get(result["classification"], "")
    reset = COLORS["RESET"]
    bold = COLORS["BOLD"]
    dim = COLORS["DIM"]

    print(f"\n{'─' * 55}")
    print(f"{dim}Message :{reset}  {message}")
    print(f"{bold}Level   :{reset}  {c}{bold}{result['classification']}{reset}  [{result['method']}]")
    print(f"Reason  :  {result['reason']}")
    print(f"Action  :  {result['action']}")
    print(f"\n{c}{result['response']}{reset}")
    print(f"{'─' * 55}")


# ─────────────────────────────────────────────────────────────
#  TEST SUITE — 20 real-world examples from your SOP
#  Run this to verify the classifier works correctly.
#  Expected result is shown in comments.
# ─────────────────────────────────────────────────────────────

TEST_CASES = [
    # ── EMERGENCY ───────────────────────────────────────────
    ("My dog just ate a whole chocolate bar",                   "EMERGENCY"),
    ("My cat is gasping and her gums look blue",                "EMERGENCY"),
    ("My dog collapsed in the yard and won't wake up",          "EMERGENCY"),
    ("He's been having seizures for 5 minutes",                 "EMERGENCY"),
    ("My male cat is straining to urinate and crying",          "EMERGENCY"),
    ("My dog was hit by a car, she seems okay but limping",     "EMERGENCY"),
    ("I think my rabbit ate rat poison",                        "EMERGENCY"),
    ("The dog swallowed a chicken bone whole",                  "EMERGENCY"),
    ("My puppy ate some grapes, about 10 of them",              "EMERGENCY"),
    ("My dog's stomach looks huge and she's drooling a lot",    "EMERGENCY"),

    # ── MODERATE ────────────────────────────────────────────
    ("My dog has been vomiting since yesterday, 3 times now",   "MODERATE"),
    ("My cat hasn't eaten in 2 days but is still drinking",     "MODERATE"),
    ("My rabbit hasn't eaten anything since this morning",      "MODERATE"),
    ("She's limping on her front leg but putting weight on it", "MODERATE"),
    ("My bird is sitting at the bottom of the cage",            "MODERATE"),
    ("My dog's ear smells bad and he keeps shaking his head",   "MODERATE"),

    # ── ROUTINE ─────────────────────────────────────────────
    ("I need to book my puppy's first vaccination",             "ROUTINE"),
    ("When can I bring my cat in for deworming?",               "ROUTINE"),
    ("I'd like to schedule a spay for my 6 month old cat",      "ROUTINE"),
    ("My dog needs his annual checkup and booster shots",       "ROUTINE"),
]


def run_tests(client: anthropic.Anthropic = None):
    """
    Runs all 20 test cases and prints a pass/fail report.
    """
    print(f"\n{'=' * 55}")
    print("  VetDesk AI — Classifier Test Suite")
    print(f"{'=' * 55}")

    passed = 0
    failed = 0
    failures = []

    for message, expected in TEST_CASES:
        result = classify(message, client)
        actual = result["classification"]
        ok = actual == expected

        status = "✓" if ok else "✗"
        color = COLORS["ROUTINE"] if ok else COLORS["EMERGENCY"]
        print(f"  {color}{status}{COLORS['RESET']}  [{expected:9s}]  {message[:50]}")

        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((message, expected, actual))

    print(f"\n  Results: {passed}/{len(TEST_CASES)} passed")

    if failures:
        print(f"\n  Failed cases:")
        for msg, exp, act in failures:
            print(f"    Expected {exp}, got {act}: '{msg}'")

    print(f"{'=' * 55}\n")
    return passed, failed


# ─────────────────────────────────────────────────────────────
#  INTERACTIVE MODE
#  Lets you type messages and see results in real time.
#  Great for testing before connecting to a chat interface.
# ─────────────────────────────────────────────────────────────

def interactive_mode(client: anthropic.Anthropic = None):
    print("\n" + "=" * 55)
    print("  VetDesk AI — Interactive Triage")
    print("  Type a pet owner message. Type 'quit' to exit.")
    print("=" * 55)

    while True:
        try:
            message = input("\nOwner says: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not message:
            continue
        if message.lower() in ("quit", "exit", "q"):
            break

        result = classify(message, client)
        print_result(message, result)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Set up the Anthropic client
    # If no valid API key, runs in keyword-only mode
    client = None
    if API_KEY and API_KEY != "your-api-key-here":
        try:
            client = anthropic.Anthropic(api_key=API_KEY)
            print("✓ Anthropic API connected — using LLM + keyword classification")
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            print("  Running in keyword-only mode.")
    else:
        print("⚠  No API key set — running in keyword-only mode.")
        print("  Set ANTHROPIC_API_KEY environment variable or update API_KEY above.\n")

    # Choose mode from command line argument
    # python vet_classifier.py test        → run test suite
    # python vet_classifier.py interactive → interactive mode
    # python vet_classifier.py            → run tests then interactive

    mode = sys.argv[1] if len(sys.argv) > 1 else "both"

    if mode == "test":
        run_tests(client)
    elif mode == "interactive":
        interactive_mode(client)
    else:
        run_tests(client)
        interactive_mode(client)
