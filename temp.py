path = r'e:\Course_resources\cse427\labs\part7_ml_algorithms_comparison.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()
content = content.replace('🏆', '').replace('⚡', '').replace('🔍', '').replace('🎯', '')
with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
