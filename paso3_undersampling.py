import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from imblearn.under_sampling import RandomUnderSampler

# Cargar dataset con ambos pasos ya aplicados
df = pd.read_csv('archive/news_category_paso2.csv')

# Conteos ANTES
before_counts = df['category'].value_counts().sort_values(ascending=False)
print('='*65)
print('PASO 3 — UNDERSAMPLING CON RandomUnderSampler')
print('='*65)
print(f'Registros ANTES: {len(df):,}')
print(f'Categorias:      {df["category"].nunique()}')
print(f'Clase minoritaria: {before_counts.idxmin()} ({before_counts.min():,})')
print(f'Clase mayoritaria: {before_counts.idxmax()} ({before_counts.max():,})')
print()

# Aplicar RandomUnderSampler
X = df[['clean_headline', 'headline_word_count', 'avg_word_length', 'year']]
y = df['category']

rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

df_balanced = X_res.copy()
df_balanced['category'] = y_res

# Conteos DESPUES
after_counts = df_balanced['category'].value_counts().sort_values(ascending=False)

print(f'Registros DESPUES: {len(df_balanced):,}')
print(f'Registros eliminados: {len(df) - len(df_balanced):,}')
print(f'Conteo por clase (todos iguales): {after_counts.min():,}')
print()

# Tabla comparativa
print('--- TABLA ANTES / DESPUES ---')
print(f'{"CATEGORIA":<28} {"ANTES":>8} {"DESPUES":>8} {"ELIMINADOS":>10}')
print('-'*58)
for cat in before_counts.index:
    b = before_counts[cat]
    a = after_counts.get(cat, 0)
    elim = b - a
    print(f'{cat:<28} {b:>8,} {a:>8,} {elim:>10,}')

# --- Gráfica de barras antes vs después ---
categories = before_counts.index.tolist()
n = len(categories)
x = np.arange(n)

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('PASO 3 — RandomUnderSampler: Antes vs Después\n(16 categorías)', fontsize=14, fontweight='bold', y=1.01)

# --- Antes ---
bars_before = axes[0].bar(x, [before_counts[c] for c in categories], color='steelblue', edgecolor='white', width=0.7)
axes[0].set_title('ANTES del Undersampling', fontsize=12, pad=10)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, rotation=55, ha='right', fontsize=9)
axes[0].set_ylabel('Número de registros', fontsize=10)
axes[0].set_xlim(-0.6, n - 0.4)
axes[0].set_ylim(0, before_counts.max() * 1.18)
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{int(val):,}'))
axes[0].grid(axis='y', linestyle='--', alpha=0.4)
for i, v in enumerate([before_counts[c] for c in categories]):
    axes[0].text(i, v + before_counts.max() * 0.01, f'{v:,}',
                 ha='center', va='bottom', fontsize=7, rotation=90)

# --- Después ---
bars_after = axes[1].bar(x, [after_counts.get(c, 0) for c in categories], color='seagreen', edgecolor='white', width=0.7)
axes[1].set_title(f'DESPUÉS del Undersampling\n(todas las clases = {after_counts.min():,})', fontsize=12, pad=10)
axes[1].set_xticks(x)
axes[1].set_xticklabels(categories, rotation=55, ha='right', fontsize=9)
axes[1].set_ylabel('Número de registros', fontsize=10)
axes[1].set_xlim(-0.6, n - 0.4)
axes[1].set_ylim(0, after_counts.min() * 1.30)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{int(val):,}'))
axes[1].grid(axis='y', linestyle='--', alpha=0.4)
for i, v in enumerate([after_counts.get(c, 0) for c in categories]):
    axes[1].text(i, v + after_counts.min() * 0.02, f'{v:,}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig('paso3_undersampling.png', dpi=150, bbox_inches='tight')
print()
print('>> Grafica guardada en paso3_undersampling.png')

# Guardar dataset balanceado para Paso 4
df_balanced.to_csv('archive/news_category_balanced.csv', index=False)
print('>> Dataset balanceado guardado en archive/news_category_balanced.csv')
