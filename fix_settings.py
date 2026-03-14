path = 'config/settings.py'
code = open(path, encoding='utf-8').read()

missing = {
    'detector_model': '"yolov9c"',
    'ocr_languages': '["en"]',
    'esrgan_scale': '4',
    'esrgan_tile': '0',
    'helmet_conf': '0.55',
    'seatbelt_conf': '0.55',
    'violation_frames': '3',
    'plate_min_area': '500',
    'fps_target': '30',
    'anonymise_faces': 'True',
    'enable_heatmap': 'False',
    'save_violations': 'True',
    'output_dir': '"outputs"',
    'camera_id': '"CCTV-001"',
    'api_port': '8000',
}

lines_to_add = []
for attr, default in missing.items():
    if attr not in code:
        lines_to_add.append(f'    {attr} = {default}')
        print(f'Adding: {attr}')
    else:
        print(f'Already exists: {attr}')

if lines_to_add:
    code = code.rstrip() + '\n' + '\n'.join(lines_to_add) + '\n'
    open(path, 'w', encoding='utf-8').write(code)
    print('Done!')
else:
    print('Nothing to add.') 
