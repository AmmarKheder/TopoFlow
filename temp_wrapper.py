# Créer un wrapper pour get_var_ids qui force la conversion

def ensure_tuple(variables):
    """Convert variables to tuple recursively"""
    if isinstance(variables, list):
        return tuple(ensure_tuple(item) if isinstance(item, list) else item for item in variables)
    return variables

# Modifier le forward du modèle PM25
import re

with open('src/model.py', 'r') as f:
    content = f.read()

# Ajouter la fonction ensure_tuple en haut
if 'def ensure_tuple' not in content:
    lines = content.split('\n')
    import_line = -1
    for i, line in enumerate(lines):
        if line.startswith('import') or line.startswith('from'):
            import_line = i
    
    wrapper_func = '''
def ensure_tuple(variables):
    """Convert variables to tuple recursively"""
    if isinstance(variables, list):
        return tuple(ensure_tuple(item) if isinstance(item, list) else item for item in variables)
    return variables
'''
    
    lines.insert(import_line + 1, wrapper_func)
    content = '\n'.join(lines)

# Remplacer tous les appels variables par ensure_tuple(variables)
content = re.sub(r'self\.climax\.forward_encoder\(([^,]+), ([^,]+), ([^)]+)\)', 
                 r'self.climax.forward_encoder(\1, \2, ensure_tuple(\3))', content)

with open('src/model.py', 'w') as f:
    f.write(content)

print("✅ Wrapper appliqué!")
