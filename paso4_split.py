import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar dataset balanceado (resultado del Paso 3)
df = pd.read_csv('archive/news_category_balanced.csv')

X = df.drop(columns=['category'])
y = df['category']

# División estratificada 70/30
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    stratify=y,
    random_state=42
)

# Reconstruir DataFrames
train_df = X_train.copy()
train_df['category'] = y_train

test_df = X_test.copy()
test_df['category'] = y_test

# Guardar archivos
train_df.to_csv('news_category_train_70.csv', index=False)
test_df.to_csv('news_category_test_30.csv',  index=False)

# --- Verificación ---
train_counts = train_df['category'].value_counts().sort_values(ascending=False)
test_counts  = test_df['category'].value_counts().sort_values(ascending=False)

print('='*65)
print('PASO 4 — DIVISION TRAIN / TEST')
print('='*65)
print(f'Total registros balanceados:  {len(df):,}')
print(f'Registros TRAIN (70%):        {len(train_df):,}')
print(f'Registros TEST  (30%):        {len(test_df):,}')
print(f'Categorias en TRAIN:          {train_df["category"].nunique()}')
print(f'Categorias en TEST:           {test_df["category"].nunique()}')
print()

print('--- CONTEOS POR CATEGORIA: TRAIN vs TEST ---')
print(f'{"CATEGORIA":<28} {"TRAIN":>7} {"TEST":>7} {"TOTAL":>7}')
print('-'*52)
for cat in train_counts.index:
    t  = train_counts.get(cat, 0)
    te = test_counts.get(cat, 0)
    print(f'{cat:<28} {t:>7,} {te:>7,} {t+te:>7,}')

print()
print(f'TOTALES:                     {len(train_df):>7,} {len(test_df):>7,} {len(df):>7,}')

# Verificar balance perfecto
min_train = train_counts.min()
max_train = train_counts.max()
min_test  = test_counts.min()
max_test  = test_counts.max()

print()
print('--- VERIFICACION DE BALANCE ---')
print(f'  TRAIN  min={min_train:,}  max={max_train:,}  ratio={max_train/min_train:.3f}x')
print(f'  TEST   min={min_test:,}   max={max_test:,}   ratio={max_test/min_test:.3f}x')
if max_train / min_train <= 1.01 and max_test / min_test <= 1.01:
    print('  >> Ambos conjuntos estan BALANCEADOS correctamente.')

print()
print('>> Guardado: news_category_train_70.csv')
print('>> Guardado: news_category_test_30.csv')
