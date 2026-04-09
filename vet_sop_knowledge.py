"""
=============================================================
  VetDesk AI — SOP Knowledge Base  (Phase 3, Part A)
  vet_sop_knowledge.py
=============================================================
  This file IS your SOP in Python form.
  It holds every piece of clinic knowledge as a list of
  plain-text chunks — the same content as your Word document
  but in a format the RAG engine can search.

  HOW TO CUSTOMISE:
    - Replace [CLINIC NAME], [PHONE], [ADDRESS] etc. with real values
    - Add or remove chunks to match your actual clinic
    - Each chunk should cover ONE topic (one price, one rule, etc.)
    - Shorter chunks = more precise answers
    - This file works for BOTH the US and Nigerian clinic editions —
      just update the content to match your version

  ADDING NEW CHUNKS:
    Just append to the CHUNKS list:
      "Your new fact here. As a complete sentence."
=============================================================
"""

# ─────────────────────────────────────────────────────────────
#  CLINIC IDENTITY
# ─────────────────────────────────────────────────────────────

CLINIC_NAME    = "[Biocity vet]"
CLINIC_PHONE   = "[1234567]"
EMERGENCY_LINE = "[24/7 765432]"
WHATSAPP       = "[567890]"
ADDRESS        = "[No 22 Health street]"
WEBSITE        = "[www.prototype.com]"

# ─────────────────────────────────────────────────────────────
#  KNOWLEDGE CHUNKS
#  Each string = one searchable unit of clinic knowledge.
#  The RAG engine splits on these and finds the best match.
# ─────────────────────────────────────────────────────────────

CHUNKS = [

    # ── IDENTITY & CONTACT ───────────────────────────────────
    f"The clinic name is {CLINIC_NAME}. The main phone number is {CLINIC_PHONE}.",
    f"The clinic address is {ADDRESS}.",
    f"The clinic website is {WEBSITE}.",
    f"The 24/7 emergency line is {EMERGENCY_LINE}. Call this number any time of day or night for life-threatening pet emergencies.",
    f"The WhatsApp number for the clinic is {WHATSAPP}. You can message us on WhatsApp for non-emergency queries.",

    # ── OPENING HOURS ─────────────────────────────────────────
    "The clinic is open Monday to Friday from 8:00 AM to 6:00 PM.",
    "The clinic is open on Saturday from 9:00 AM to 4:00 PM.",
    "The clinic is closed on Sundays and all public holidays. Emergency line stays active.",
    "Sunday hours: the clinic is closed on Sunday. For Sunday emergencies call the 24/7 emergency line.",
    "Are we open on Sunday? No. The clinic is closed every Sunday. Please call the emergency line for urgent Sunday cases.",
    "Last appointments are taken 30 minutes before closing time.",

    # ── ANIMALS TREATED ───────────────────────────────────────
    "We treat dogs of all breeds, including mixed breeds.",
    "We treat cats of all breeds, including domestic shorthair, longhair, Persian, Maine Coon, and Siamese.",
    "We treat rabbits of all breeds as exotic small animals.",
    "We treat pet birds including parrots, cockatiels, canaries, lovebirds, budgerigars, and conures.",
    "We do not treat commercial livestock, venomous reptiles, or large wild animals. For these, please contact a large animal vet or wildlife rehabilitator.",
    "Backyard chickens are seen on a case-by-case basis. Please call ahead.",

    # ── SERVICES ─────────────────────────────────────────────
    "We offer consultations and physical examinations for all patients.",
    "We offer vaccinations for dogs and cats including core and non-core vaccines.",
    "We offer deworming and parasite prevention for dogs, cats, rabbits, and birds.",
    "We perform spay and neuter surgeries for dogs and cats.",
    "We perform soft tissue surgery including wound repair, mass removal, and foreign body removal.",
    "We offer dental cleaning under anaesthesia for dogs and cats.",
    "We offer diagnostic services including blood tests, urinalysis, and X-rays.",
    "We offer microchipping and ISO-standard microchip registration.",
    "We issue health certificates and Certificates of Veterinary Inspection for travel.",
    "We offer hospitalisation and post-surgical monitoring.",
    "We offer nail trims and minor grooming services.",

    # ── PRICING — CONSULTATIONS ──────────────────────────────
    "A wellness exam or consultation costs between $50 and $100 depending on the vet and complexity.",
    "An emergency consultation fee applies for after-hours or urgent walk-in cases.",

    # ── PRICING — VACCINES ───────────────────────────────────
    "The DA2PP vaccine for dogs costs between $20 and $45 per dose.",
    "The rabies vaccine costs between $15 and $35 and includes the vaccination certificate.",
    "The Bordetella kennel cough vaccine costs between $20 and $45.",
    "The Lyme disease vaccine costs between $25 and $50 per dose.",
    "The FVRCP cat vaccine costs between $20 and $45 per dose.",
    "The FeLV feline leukemia vaccine costs between $20 and $50 per dose.",

    # ── PRICING — PARASITE CONTROL ───────────────────────────
    "Heartworm testing using the 4Dx test costs between $45 and $80.",
    "Monthly heartworm prevention products cost between $8 and $20 per month.",
    "Monthly flea and tick prevention products cost between $15 and $30 per month.",
    "Deworming costs between $15 and $40 depending on the pet's weight and the product used.",

    # ── PRICING — SURGERY ────────────────────────────────────
    "Spay surgery for a cat costs between $200 and $500 including anaesthesia and post-op care.",
    "Spay surgery for a small or medium dog costs between $300 and $800 depending on weight.",
    "Neuter surgery for a cat costs between $150 and $400.",
    "Neuter surgery for a small or medium dog costs between $200 and $600 depending on weight.",
    "Dental cleaning under anaesthesia costs between $300 and $800. Tooth extractions are charged separately.",

    # ── PRICING — DIAGNOSTICS ────────────────────────────────
    "A complete blood count and chemistry panel costs between $80 and $200.",
    "An X-ray radiograph costs between $75 and $250 per view or series.",
    "Urinalysis costs between $30 and $60.",

    # ── PRICING — OTHER ──────────────────────────────────────
    "Microchip implantation costs between $25 and $60 and includes registration.",
    "A health certificate or Certificate of Veterinary Inspection for travel costs between $75 and $200 depending on the destination.",
    "Hospitalisation costs between $50 and $200 per day, not including medications.",
    "Nail trimming costs between $15 and $30.",
    "All prices are estimates. Exact costs are confirmed at the appointment.",
    "We accept major credit cards, cash, and CareCredit for qualified clients.",

    # ── VACCINATIONS — DOG SCHEDULE ──────────────────────────
    "Puppies receive the DA2PP vaccine at 6 to 8 weeks, 10 to 12 weeks, and 14 to 16 weeks old.",
    "Adult dogs receive a DA2PP booster one year after the puppy series, then every three years.",
    "Dogs receive their first rabies vaccine between 12 and 16 weeks old.",
    "Dogs need a rabies booster one year after the first vaccine, then every one to three years depending on state law.",
    "Rabies vaccination is legally required for dogs in all 50 US states.",
    "The Bordetella vaccine is recommended annually or every six months for dogs that visit parks, boarding, or dog shows.",
    "The Lyme vaccine is recommended annually for dogs in tick-endemic regions.",
    "The canine influenza vaccine is recommended for social dogs that visit boarding, dog parks, or dog shows.",
    "The leptospirosis vaccine is recommended annually for dogs with outdoor or water exposure.",

    # ── VACCINATIONS — CAT SCHEDULE ──────────────────────────
    "Kittens receive the FVRCP vaccine at 6 to 8 weeks, 10 to 12 weeks, and 14 to 16 weeks old.",
    "Adult cats receive an FVRCP booster one year after the kitten series, then every three years.",
    "Cats receive their first rabies vaccine between 12 and 16 weeks old.",
    "The FeLV vaccine is recommended for outdoor cats or cats in multi-cat households.",
    "We use non-adjuvanted rabies vaccines for cats to reduce injection site reaction risk.",

    # ── DEWORMING & PARASITE CONTROL ─────────────────────────
    "Puppies should be dewormed at 2, 4, 6, and 8 weeks old, then monthly until 6 months.",
    "Adult dogs should be dewormed based on annual fecal exam results, or every 3 to 6 months for outdoor dogs.",
    "Kittens should be dewormed at 4, 6, and 8 weeks old, then monthly until 6 months.",
    "Adult cats should be dewormed based on annual fecal exam results.",
    "Year-round heartworm prevention is recommended for all dogs and cats.",
    "Dogs should have an annual 4Dx blood test to screen for heartworm, Lyme, Ehrlichia, and Anaplasma.",

    # ── SURGERY & FASTING RULES ──────────────────────────────
    "Dogs must not eat for 8 to 12 hours before surgery. Water can be given until 4 hours before.",
    "How long to fast a dog before surgery: no food for 8 to 12 hours, no water for 4 hours before the procedure.",
    "Cats must not eat for 8 to 12 hours before surgery. Water can be given until 4 hours before.",
    "Rabbits must NOT be fasted before surgery. Fasting rabbits causes dangerous GI stasis. Do not withhold food or water from a rabbit before a procedure.",
    "Birds should only be fasted for 2 to 4 hours before surgery. Never withhold water from a bird.",
    "Pre-surgical bloodwork is recommended for all patients over 7 years old.",

    # ── HOSPITALISATION ───────────────────────────────────────
    "Visiting hours for hospitalised patients are from 12:00 PM to 2:00 PM daily, subject to the patient's condition.",
    "Owners of hospitalised pets receive daily updates from our team.",
    "Discharge is authorised by the attending vet only.",

    # ── TRIAGE GUIDANCE ───────────────────────────────────────
    "Signs of a pet emergency include: difficulty breathing, blue or pale gums, collapse, seizures, suspected poisoning, inability to urinate especially in male cats, severe trauma, and uncontrolled bleeding.",
    "Chocolate, grapes, raisins, xylitol, onions, garlic, ibuprofen, acetaminophen, macadamia nuts, antifreeze, and rat poison are all toxic to dogs and cats. Any ingestion is an emergency.",
    "A male cat that is straining to urinate and producing no urine has a blocked bladder. This is a life-threatening emergency.",
    "Rabbits that stop eating for more than 12 hours may be developing GI stasis, which can be fatal. This requires same-day veterinary attention.",
    "A bird sitting at the bottom of its cage, puffed up, or not eating should be seen by a vet within 24 hours.",

    # ── APPOINTMENTS ──────────────────────────────────────────
    "To book an appointment, call us at the main clinic number or message us on WhatsApp.",
    "Morning appointment slots are at 9:00 AM, 10:00 AM, and 11:00 AM.",
    "Afternoon appointment slots are at 1:00 PM, 2:00 PM, 3:00 PM, and 4:00 PM.",
    "Walk-in patients are welcome but booked appointments are seen first.",
    "For emergencies, walk-in patients are always prioritised regardless of appointment schedule.",
    "Please arrive 10 minutes before your appointment with any previous vaccination records.",
    "To book an appointment we need: your pet's name, species and breed, approximate age, your name, your phone number, and the reason for the visit.",

    # ── TRAVEL CERTIFICATES ───────────────────────────────────
    "A health certificate is required for air travel and interstate travel with a pet in the United States.",
    "Health certificates must be issued by a USDA-accredited veterinarian within 10 days of travel.",
    "International travel requires a USDA-endorsed health certificate and may require additional testing or treatment. Please contact us at least two weeks before travel.",

    # ── SPAY & NEUTER ─────────────────────────────────────────
    "The recommended age to spay a female dog is between 6 and 12 months depending on breed size.",
    "The recommended age to neuter a male dog is between 6 and 12 months depending on breed size.",
    "Cats are typically spayed or neutered at 5 to 6 months old.",
    "Spaying and neutering reduces the risk of certain cancers and prevents unwanted pregnancies.",

    # ── MICROCHIPPING ─────────────────────────────────────────
    "Microchipping involves a small ISO-standard chip injected under the skin between the shoulder blades.",
    "A microchip is a permanent form of pet identification and is recommended for all dogs and cats.",
    "After microchipping we register the chip with a national database and provide the owner with the chip number.",

    # ── GENERAL FAQs ─────────────────────────────────────────
    "We do not diagnose or prescribe medications over the phone or chat. A physical examination is required.",
    "If your pet has a dental issue, bad breath, or broken tooth, a dental check is recommended. Cleaning is done under anaesthesia.",
    "Post-surgical follow-up appointments are included in the surgical fee for the first two weeks.",
    "We recommend annual wellness exams for all pets regardless of age.",
    "Senior pets over 7 years old benefit from twice-yearly wellness exams.",
    "Home visits are available on a case-by-case basis for mobility-limited patients. Call to arrange.",
]
