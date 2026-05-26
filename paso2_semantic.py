import pandas as pd

df = pd.read_csv('archive/news_category_ml_ready.csv')

# --- PASO 1: aplicar primero el cleanup de duplicados ---
CATEGORY_CLEANUP = {
    'HEALTHY LIVING':  'WELLNESS',
    'PARENTS':         'PARENTING',
    'ARTS':            'ARTS & CULTURE',
    'CULTURE & ARTS':  'ARTS & CULTURE',
    'THE WORLDPOST':   'WORLD NEWS',
    'WORLDPOST':       'WORLD NEWS',
    'STYLE':           'STYLE & BEAUTY',
    'BUSINESS':        'BUSINESS & MONEY',
    'MONEY':           'BUSINESS & MONEY',
    'IMPACT':          'SOCIAL IMPACT',
    'GOOD NEWS':       'SOCIAL IMPACT',
    'GREEN':           'ENVIRONMENT',
    'TASTE':           'FOOD & DRINK',
    'FIFTY':           'WELLNESS',
    'COLLEGE':         'EDUCATION',
}
df['category'] = df['category'].replace(CATEGORY_CLEANUP)

# Estado ANTES del paso 2
before = df['category'].value_counts().reset_index()
before.columns = ['category', 'count_before']

# --- PASO 2: fusión semántica ---
CATEGORY_SEMANTIC = {
    'U.S. NEWS':       'POLITICS',
    'WORLD NEWS':      'POLITICS',
    'QUEER VOICES':    'IDENTITY VOICES',
    'BLACK VOICES':    'IDENTITY VOICES',
    'LATINO VOICES':   'IDENTITY VOICES',
    'SCIENCE':         'SCIENCE & TECH',
    'TECH':            'SCIENCE & TECH',
    'ENVIRONMENT':     'SCIENCE & TECH',
    'ARTS & CULTURE':  'ENTERTAINMENT',
    'COMEDY':          'ENTERTAINMENT',
    'PARENTING':       'FAMILY & RELATIONSHIPS',
    'RELATIONSHIPS':   'FAMILY & RELATIONSHIPS',
    'WEDDINGS':        'FAMILY & RELATIONSHIPS',
    'DIVORCE':         'FAMILY & RELATIONSHIPS',
    'WOMEN':           'FAMILY & RELATIONSHIPS',
    'FOOD & DRINK':    'WELLNESS',
    'HOME & LIVING':   'WELLNESS',
}
df['category'] = df['category'].replace(CATEGORY_SEMANTIC)

# Estado DESPUES del paso 2
after = df['category'].value_counts().reset_index()
after.columns = ['category', 'count_after']

# Calcular ratio desbalance
min_count = after['count_after'].min()
max_count = after['count_after'].max()
min_cat   = after.loc[after['count_after'].idxmin(), 'category']
max_cat   = after.loc[after['count_after'].idxmax(), 'category']
ratio     = max_count / min_count

print('='*65)
print('TABLA ANTES/DESPUES — PASO 2: FUSION SEMANTICA')
print('='*65)
print(f'Categorias ANTES (tras Paso 1): {before["category"].nunique()}')
print(f'Categorias DESPUES:             {after["category"].nunique()}')
print(f'Total registros (sin cambio):   {len(df):,}')
print()

# Tabla antes
print('--- ESTADO ANTES DEL PASO 2 (29 categorias) ---')
print(f'{"CATEGORIA":<28} {"CONTEO":>8}')
print('-'*38)
for _, row in before.iterrows():
    print(f'{row["category"]:<28} {row["count_before"]:>8,}')

print()
print('--- ESTADO DESPUES DEL PASO 2 ---')
print(f'{"CATEGORIA":<28} {"CONTEO":>8}')
print('-'*38)
for _, row in after.iterrows():
    print(f'{row["category"]:<28} {row["count_after"]:>8,}')

print()
print('--- RATIO DE DESBALANCE ---')
print(f'  Clase MAXIMA: {max_cat} ({max_count:,})')
print(f'  Clase MINIMA: {min_cat} ({min_count:,})')
print(f'  Ratio max/min: {ratio:.2f}x')

# Guardar estado intermedio para pasos siguientes
df.to_csv('archive/news_category_paso2.csv', index=False)
print()
print('>> Dataset guardado en archive/news_category_paso2.csv')
