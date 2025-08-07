
import os
import openai
import json
import base64

# 1. Set your API key in your environment:
#    export OPENAI_API_KEY="sk-…" (or equivalent on Windows)
openai.api_key = os.getenv("OPENAI_API_KEY")

# 2. Define the system prompt to initialize your LITIGATION OS vFUSION.ULTRA
system_prompt = """
You are LITIGATION OS vFUSION.ULTRA, Level 9999 in ETERNITY+ MODE with Truth Lock.
All modules are live (MemoryCrawler, Canon Enforcer, Statute–Filing Matcher,
AHIS, JRAE, WTNC, TFHM, MFAG, VeilPiercer, etc.). Data mounts:
• /mnt/data/google_drive  (rclone Google Drive mirror)
• /mnt/data/F_drive       (F:\\ Master Knowledge Engine)
• /mnt/data/D_drive       (D:\\ Test evidence scan)
• /mnt/data/Z_drive       (Z:\\ Code Keeper & doc ingestion)
GUI.exe litigation command center online. Use Michigan Court Rules (MCR),
Michigan Compiled Laws (MCL), and Benchbook authority. Enforce
no-placeholder, final, court-ready output. Triple-verify all citations.
Output must be a base64-encoded `.docx` in JSON.
"""

# 3. Define the user prompt describing exactly what you need
user_prompt = """
Task: 
1) Recursively scan the drives for all relevant evidence in case 2025-002760-CZ.
2) Generate a Motion to Strike Lease, Convert Motion Hearing to Damages Hearing,
   Impose Sanctions, and Rebut Defendant’s Motion for Sanctions & Filing Restrictions.

Requirements:
- Caption: STATE OF MICHIGAN, 14th CIRCUIT COURT, COUNTY OF MUSKEGON; Case No. 2025-002760-CZ.
- Plaintiff: Andrew J. Pigors; Defendants: Shady Oaks MHP LLC, Homes of America LLC, etc.
- Legal authority: cite MCR 2.114(D), MCR 2.115(B)–(C), MCR 1.109(E)(5), MCL 600.2591, plus relevant Benchbook entries.
- Exhibits A–D: include cover page titles, authentication language per Michigan rules.
- Include signature block and Certificate of Service formatted per SCAO standards.
- Output JSON:
  {
    "filename": "Motion_to_Strike_Lease.docx",
    "content_base64": "<BASE64-ENCODED DOCX>"
  }
Ensure the `.docx` is MiFile-ready and contains no placeholders.
"""

# 4. Call the GPT-4 Turbo (gpt-4o) model to generate the document
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ],
    temperature=0.0,
    max_tokens=20000,
)

# 5. Parse the model’s JSON output and write the .docx
res = response.choices[0].message.content
doc = json.loads(res)
doc_bytes = base64.b64decode(doc["content_base64"])

with open(doc["filename"], "wb") as f:
    f.write(doc_bytes)

print(f"✅ {doc['filename']} saved successfully at {os.getcwd()}")
