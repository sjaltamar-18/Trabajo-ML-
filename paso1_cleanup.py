import pandas as pd

df = pd.read_csv('archive/news_category_ml_ready.csv')

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

before = df['category'].value_counts().reset_index()
before.columns = ['category', 'count_before']

df['category'] = df['category'].replace(CATEGORY_CLEANUP)

after = df['category'].value_counts().reset_index()
after.columns = ['category', 'count_after']

print('='*65)
print('TABLA ANTES/DESPUES — PASO 1: FUSION DE DUPLICADOS OBVIOS')
print('='*65)
print(f'Categorias ANTES:  {before["category"].nunique()}')
print(f'Categorias DESPUES: {after["category"].nunique()}')
print(f'Total registros (sin cambio): {len(df)}')
print()
print('--- CATEGORIAS FUSIONADAS (origen -> destino) ---')
for src, dst in CATEGORY_CLEANUP.items():
    src_count = before[before['category']==src]['count_before'].values
    src_count = src_count[0] if len(src_count) > 0 else 0
    print(f'  {src:<20} ({src_count:>5}) -> {dst}')

print()
print('--- CONTEOS POR CATEGORIA DESPUES (ordenado desc) ---')
print(f'{"CATEGORIA":<25} {"CONTEO":>8}')
print('-'*35)
for _, row in after.iterrows():
    print(f'{row["category"]:<25} {row["count_after"]:>8,}')
