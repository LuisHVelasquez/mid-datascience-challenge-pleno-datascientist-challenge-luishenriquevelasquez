##### Carregando o banco de dados ----
import pandas as pd 

df = pd.read_csv(r'C:\Users\luish\Stefanini\Desafio IHM Stefanini\MiningProcess_Flotation_Plant_Database.csv',
                 parse_dates=['date'], # Definindo qual coluna apresenta dados tipo "data"
                 decimal=",") # Indicando o separados de decimais para evitar problemas futuros


##### Modelo BETA (GLM) ----

### Ajustando as variáveis necessárias para o modelo

# Indicador relacionado ao Fluxo de Ar
df['FluxoAr_indicador'] = df[['Flotation Column 01 Air Flow',
                              'Flotation Column 02 Air Flow',
                              'Flotation Column 03 Air Flow',
                              'Flotation Column 04 Air Flow',
                              'Flotation Column 05 Air Flow',
                              'Flotation Column 06 Air Flow',
                              'Flotation Column 07 Air Flow'
                            ]].mean(axis=1) 

# Indicador relacionado ao Nível de Espuma                          
df['NivelEspuma_indicador'] = df[['Flotation Column 01 Level',
                                  'Flotation Column 02 Level',
                                  'Flotation Column 03 Level',
                                  'Flotation Column 04 Level',
                                  'Flotation Column 05 Level',
                                  'Flotation Column 06 Level',
                                  'Flotation Column 07 Level'
                                ]].mean(axis=1)  

Var_Resposta_porc = df['% Silica Concentrate']/100 # Dividindo a variável por 100 para se ajustar ao intervalo (0,1)
print(Var_Resposta_porc.head(10))
import numpy as np
Var_Explicativas_ind = df.iloc[:,np.r_[1:8, 24:len(df.columns)]]
print(Var_Explicativas_ind.columns)



### Separando o banco de dados em Training and Test Set
from sklearn.model_selection import train_test_split
import random
random.seed(123) # Definindo a semente aleatória para os resultados serem os mesmos sempre que rodarmos os códigos
x_treino,x_test,y_treino,y_test=train_test_split(Var_Explicativas_ind,Var_Resposta_porc,test_size=0.25) # 25% para o teste e 75% para o treinamento

# Base de Treinamento
print(Var_Explicativas_ind.head(5)) # ordem original dos dados
print(x_treino.head(5)) # os dados estão de fato aleatorizados
print(x_treino.shape) # 553089 linhas de 737453 (75%)

# Base Teste
print(x_test.head(5)) # os dados estão de fato aleatorizados
print(x_test.shape) # 184364 linhas de 737453 (25%)


### Treinando os dados
import statsmodels.api as sm
from statsmodels.othermod.betareg import BetaModel
x_treino_const = sm.add_constant(x_treino) # Adicionando uma contante (B0) a regressão
print(x_treino_const)
mod_treino = BetaModel(y_treino, x_treino_const) # Link padrão: logit
resultado_treino = mod_treino.fit()
print(resultado_treino.summary())


### Realizando predições e calculando as métricas de qualidade do modelo
x_test_const = sm.add_constant(x_test) # Adicionando uma constante ao modelo para não gerar problemas futuros
print(x_test_const.head(5))
y_pred = resultado_treino.predict(x_test_const)
print(y_pred)
print(y_pred.describe()) # verificando as métricas da predição: média = 0.023272, min = 0.012549, max = 0.043000

# Verificando o VIF do modelo atualizado (sem "% Iron Feed" e com indicadores)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = add_constant(x_treino)
print(pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns))

# MAE
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(y_test, y_pred)
print(MAE) # 0.0085 (0,85%)

# R² Ajustado
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2_adj = 1 - (((1 - r2)*(len(x_treino_const)-1))/(len(x_treino_const)-Var_Explicativas_ind.shape[1]-1)) # Cálculo manual do R² Ajustado
print(r2) # 11,11%
print(r2_adj) # 11,11% (aproximadamente o mesmo do R²)

# MAPE
from sklearn.metrics import mean_absolute_percentage_error
MAPE = mean_absolute_percentage_error(y_test, y_pred)
print(MAPE) # 0.419 (41,9%)

# RMSE/MSE
from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y_test, y_pred, squared=False)
print(RMSE) # 0.0106 (1,06%)
MSE = mean_squared_error(y_test, y_pred, squared=True) # É a raiz quadrada do RMSE
print(MSE) # 0.0001 (0,1%)

