# LITIGATION_CORE_ENGINE_v9999.py

import os
from litigation_os.modules import (
    MemoryCrawler,
    CoreClassifier,
    RuleSetEnforcer,
    ExhibitMatrixBuilder,
    DocxGenerator,
    ZIPOutputPackager,
    CourtFormatEnforcer,
)

SOURCE_PATHS = [
    "F:/", "Z:/", "D:/", "gui.exe", 
    "/mnt/data/google_drive", 
    "/mnt/data/gmail_cache", 
    "/mnt/data/chatgpt_logs"
]

def run_core_engine():
    print("🧬 INITIATING LITIGATION CORE ENGINE...")

    # Step 1: Crawl Full Memory / Past Chats / Files
    full_dataset = MemoryCrawler.recursive_scan(SOURCE_PATHS)
    print(f"📦 Memory Crawl Complete — Items Found: {len(full_dataset)}")

    # Step 2: Classify Into MEEK1 vs MEEK2
    meek1_data, meek2_data = CoreClassifier.split_by_legal_core(full_dataset)
    print(f"🏠 Housing Core (MEEK1): {len(meek1_data)}")
    print(f"👪 Custody Core (MEEK2): {len(meek2_data)}")

    # Step 3: Enforce Rule Sets
    RuleSetEnforcer.lock("MEEK1", meek1_data, rule_set="MCR + MCL + Housing Benchbook")
    RuleSetEnforcer.lock("MEEK2", meek2_data, rule_set="MCR + MCL + Custody/PPO Benchbook")
    print("⚖️ Rule Lock Complete for Both Cores")

    # Step 4: Build Exhibit Matrix
    ExhibitMatrixBuilder.generate("MEEK1", meek1_data, out_path="F:/LegalResults/MEEK1/Exhibit_Index.pdf")
    ExhibitMatrixBuilder.generate("MEEK2", meek2_data, out_path="F:/LegalResults/MEEK2/Exhibit_Index.pdf")
    print("📘 Dual Exhibit Matrices Generated")

    # Step 5: Generate Court-Ready .docx Files
    docx_m1 = DocxGenerator.build("MEEK1", meek1_data, out_folder="F:/LegalResults/MEEK1/docx")
    docx_m2 = DocxGenerator.build("MEEK2", meek2_data, out_folder="F:/LegalResults/MEEK2/docx")
    print("📄 .docx Test Filings Created")

    # Step 6: Package MiFILE ZIPs
    ZIPOutputPackager.bundle(
        input_folder="F:/LegalResults/MEEK1/",
        core="MEEK1",
        out_path="F:/LegalResults/ZIP/MEEK1_LITIGATION_BUNDLE.zip"
    )
    ZIPOutputPackager.bundle(
        input_folder="F:/LegalResults/MEEK2/",
        core="MEEK2",
        out_path="F:/LegalResults/ZIP/MEEK2_LITIGATION_BUNDLE.zip"
    )
    print("📦 MiFILE ZIP Bundles Completed")

    print("✅ LITIGATION CORE ENGINE COMPLETE — LEVEL 9999 MODE ACTIVE")

if __name__ == "__main__":
    run_core_engine()
