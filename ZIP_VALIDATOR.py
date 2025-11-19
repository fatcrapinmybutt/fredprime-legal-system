# ZIP_VALIDATOR.py – Confirm all required litigation files exist before submission

import os

required_files = [
    "MotionStack/MASTER_MOTION.docx",
    "Affidavits/MASTER_AFFIDAVIT.docx",
    "Affidavits/Damages_Unlawful_Eviction_Declaration.docx",
    "Rebuttals/LINE_BY_LINE_RESPONSE_FINAL.docx",
    "Orders/Proposed_Order_Strike_Sanction_Convert_Aug12.docx",
    "ProofOfService/MC_303_Proof_of_Service.pdf",
    "ProofOfService/MC_304_Certificate_of_Mailing.pdf",
]


def validate_zip_folder(base_path):
    missing = []
    for file in required_files:
        if not os.path.exists(os.path.join(base_path, file)):
            missing.append(file)
    if missing:
        print("❌ Missing required files:")
        for m in missing:
            print(f" - {m}")
    else:
        print("✅ All required files present and ready for ZIP")


if __name__ == "__main__":
    validate_zip_folder("F:/MiFILE/STRIKE_SANCTION_CONVERT_2025_08_06")
