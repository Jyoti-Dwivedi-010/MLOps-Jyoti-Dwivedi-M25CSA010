#!/usr/bin/env python3
"""
Convert RTF files to plain text by extracting both Unicode and plain text
"""
import re

def parse_rtf(content, is_reference=False):
    """Parse RTF content and extract text"""
    if is_reference:
        # Reference file contains plain English text with \ for newlines
        lines = content.split('\\')
        result = []
        found_start = False
        for i, line in enumerate(lines):
            if 'reference_english' in line or found_start:
                found_start = True
                # Clean up RTF artifacts
                line = re.sub(r'[{}]', '', line)
                line = re.sub(r'f0|fs24|cf0|par|pard|tx\d+|ansicpg\d+|ansi|.*?cocoartf.*', '', line)
                line = line.strip()
                # Skip empty lines and pure numbers/RTF commands
                if line and not line.isdigit() and len(line) > 2:
                    if not any(x in line for x in ['rtf1', 'paperw', 'margl', 'expandedcolortbl']):
                        result.append(line)
        
        text = '\n'.join(result)
        # Clean up extra whitespace
        text = re.sub(r'\n\s*\n', '\n', text).strip()
        return text
    else:
        # Input file contains Bengali text with \uNNNN patterns
        # Use regex to find all unicode code points
        pattern = r'\\u(\d+)'
        matches = re.findall(pattern, content)
        
        result = []
        for code_point_str in matches:
            try:
                code_point = int(code_point_str)
                result.append(chr(code_point))
            except:
                pass
        
        text = ''.join(result)
        # Return the complete Bengali text as-is
        # (actual word breaks would need linguistic processing)
        return text.strip()

# Convert input file (Bengali text)
with open('input.rtf', 'r', encoding='utf-8') as f:
    content = f.read()
    
input_text = parse_rtf(content, is_reference=False)

with open('input.txt', 'w', encoding='utf-8') as f:
    f.write(input_text)

# Convert reference file (English text) 
with open('reference.rtf', 'r', encoding='utf-8') as f:
    content = f.read()
    
ref_text = parse_rtf(content, is_reference=True)

with open('reference.txt', 'w', encoding='utf-8') as f:
    f.write(ref_text)

print("✓ Converted input.rtf to input.txt")
print("✓ Converted reference.rtf to reference.txt")
print(f"Input text: {len(input_text)} characters, {len(input_text.split(chr(10)))} lines")
print(f"Reference text: {len(ref_text)} characters, {len(ref_text.split(chr(10)))} lines")
print("\nFirst 2 lines of input:")
for line in input_text.split('\n')[:2]:
    print(f"  {line[:100]}")
print("\nFirst 2 lines of reference:")
for line in ref_text.split('\n')[:2]:
    print(f"  {line[:100]}")
