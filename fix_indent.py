path = r'core\plate_recogniser.py'
code = open(path, encoding='utf-8').read()

lines = code.split('\n')
fixed = []
i = 0
while i < len(lines):
    line = lines[i]
    fixed.append(line)
    # Detect empty function body (def line followed by non-indented or empty)
    if line.strip().startswith('def ') and line.strip().endswith(':'):
        # Check next non-empty line
        j = i + 1
        while j < len(lines) and lines[j].strip() == '':
            j += 1
        if j >= len(lines) or (lines[j].strip() != '' and not lines[j].startswith('    ')):
            # Function has no body — add pass
            indent = len(line) - len(line.lstrip()) + 4
            fixed.append(' ' * indent + 'pass')
            print(f'Added pass to empty function at line {i+1}: {line.strip()[:50]}')
    i += 1

code = '\n'.join(fixed)
open(path, 'w', encoding='utf-8').write(code)
print('Done!')