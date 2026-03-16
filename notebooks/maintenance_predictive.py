import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Colonnes du dataset
columns = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

# Chargement
train = pd.read_csv('data/CMaps/train_FD001.txt', sep='\s+', header=None, engine='python')
train.columns = columns

# Calcul du RUL
max_cycles = train.groupby('unit')['cycle'].max().reset_index()
max_cycles.columns = ['unit', 'max_cycle']
train = train.merge(max_cycles, on='unit')
train['RUL'] = train['max_cycle'] - train['cycle']
train.drop(columns=['max_cycle'], inplace=True)

# Vérification
print("Shape :", train.shape)
print("\nRUL min / max :", train['RUL'].min(), "/", train['RUL'].max())
print("\nAperçu :")
print(train[['unit', 'cycle', 'RUL']].head(10)) 
# Visualisation RUL par moteur
plt.figure(figsize=(12, 5))
for unit in train['unit'].unique()[:10]:  # on affiche 10 moteurs
    data = train[train['unit'] == unit]
    plt.plot(data['cycle'], data['RUL'], alpha=0.6)

plt.xlabel('Cycle')
plt.ylabel('RUL (cycles restants)')
plt.title('Remaining Useful Life — 10 moteurs')
plt.tight_layout()
plt.savefig('images/RUL_par_moteur.png', dpi=150)
plt.show()
print("Graphique sauvegardé dans images/")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Features et cible
features = [f's{i}' for i in range(1, 22)]
X = train[features]
y = train['RUL']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nRésultats du modèle :")
print(f"MAE  : {mae:.2f} cycles")
print(f"RMSE : {rmse:.2f} cycles")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Features et cible
features = [f's{i}' for i in range(1, 22)]
X = train[features]
y = train['RUL']

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nRésultats du modèle :")
print(f"MAE  : {mae:.2f} cycles")
print(f"RMSE : {rmse:.2f} cycles") 

# Visualisation prédictions vs réalité
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
plt.plot([0, 300], [0, 300], 'r--', label='Prédiction parfaite')
plt.xlabel('RUL réel (cycles)')
plt.ylabel('RUL prédit (cycles)')
plt.title('Prédictions vs Réalité — Random Forest')
plt.legend()
plt.tight_layout()
plt.savefig('images/predictions_vs_realite.png', dpi=150)
plt.show()
print("Graphique sauvegardé !")