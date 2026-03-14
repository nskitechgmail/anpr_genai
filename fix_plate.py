path = 'config/settings.py'
code = open(path, encoding='utf-8').read()

tweaks = {
    'plate_min_area': ('500', '200'),
    'conf_thresh':    ('0.40', '0.20'),
    'iou_thresh':     ('0.45', '0.40'),
}

for attr, (old_val, new_val) in tweaks.items():
    if f'{attr} = {old_val}' in code:
        code = code.replace(f'{attr} = {old_val}', f'{attr} = {new_val}')
        print(f'Updated {attr}: {old_val} -> {new_val}')
    else:
        print(f'Skipped {attr} (not found with value {old_val})')

open(path, 'w', encoding='utf-8').write(code)
print('Done!')