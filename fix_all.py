"""
fix_all.py — Run this once to fix all remaining issues.
Place in your project root and run: python fix_all.py
"""
import os

# ── Fix 1: settings.py — add missing attributes ───────────────────────────
settings_path = os.path.join('config', 'settings.py')
code = open(settings_path, encoding='utf-8').read()

missing = {
    'detector_model':  '"yolov9c"',
    'iou_thresh':      '0.45',
    'ocr_languages':   '["en"]',
    'esrgan_scale':    '4',
    'esrgan_tile':     '0',
    'helmet_conf':     '0.55',
    'seatbelt_conf':   '0.55',
    'violation_frames':'3',
    'plate_min_area':  '200',
    'fps_target':      '30',
    'anonymise_faces': 'True',
    'enable_heatmap':  'False',
    'save_violations': 'True',
    'output_dir':      '"outputs"',
    'camera_id':       '"CCTV-001"',
    'api_port':        '8000',
    'use_genai':       'True',
    'conf_thresh':     '0.25',
    'device':          '"cpu"',
    'source':          '"0"',
}

added = []
for attr, default in missing.items():
    if attr not in code:
        code = code.rstrip() + f'\n    {attr} = {default}'
        added.append(attr)

if added:
    open(settings_path, 'w', encoding='utf-8').write(code)
    print(f'settings.py — added: {", ".join(added)}')
else:
    print('settings.py — all attributes already present')

# ── Fix 2: main.py — use ANPRDashboard(settings) not (pipeline, settings) ─
main_path = 'main.py'
main_code = open(main_path, encoding='utf-8').read()

if 'ANPRDashboard(pipeline, settings)' in main_code:
    main_code = main_code.replace(
        'ANPRDashboard(pipeline, settings)',
        'ANPRDashboard(settings)'
    )
    open(main_path, 'w', encoding='utf-8').write(main_code)
    print('main.py — fixed ANPRDashboard call')
else:
    print('main.py — ANPRDashboard call already correct')

print('\nAll fixes applied! Now run: python main.py --no-genai')
