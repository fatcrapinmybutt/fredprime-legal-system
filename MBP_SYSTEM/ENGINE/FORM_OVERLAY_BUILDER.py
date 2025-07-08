import os
from docx import Document


def fill_form(template_path, output_path, fields):
    doc = Document(template_path)
    for p in doc.paragraphs:
        for key, val in fields.items():
            if key in p.text:
                p.text = p.text.replace(key, val)
    doc.save(output_path)
    print(f'Form generated: {output_path}')


if __name__ == '__main__':
    fields = {
        '[NAME]': 'Andrew J Pigors',
        '[CASE]': '2024-0000001507-DC',
        '[DATE]': '07/07/2025',
        '[MOTION]': 'Custody Modification',
    }
    fill_form(
        'MBP_SYSTEM/FORMS/MC_230_TEMPLATE.docx',
        'MBP_SYSTEM/VFS/GENERATED_ZIPS/MC_230_Filled.docx',
        fields,
    )
