import os
def verify_key(name):
    val = os.getenv(name)
    if not val or len(val) < 20:
        raise ValueError(f"🔒 {name} missing or too short")
    print(f"✅ {name} loaded")
verify_key("OPENAI_KEY")
verify_key("MBP_LICENSE")
