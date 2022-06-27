##### Carregando o banco de dados ----
import pandas as pd # Instalando: "pip install pandas"

df = pd.read_csv(r'C:\Users\luish\Stefanini\Desafio IHM Stefanini\MiningProcess_Flotation_Plant_Database.csv',
                 parse_dates=['date'], # Definindo qual coluna apresenta dados tipo "data"
                 decimal=",") # Indicando o separados de decimais para evitar problemas futuros



##### Regressão Linear Multivariada ----

### Construindo uma regressão linear com todas as variáveis do banco original (com exceção da "date" e "% Iron Concentrate") ----

# Vamos ignorar a data por enquanto e a variável "% Iron Concentrate", pois ela é um produto do final da produção e está altamente correlacionado com nossa var. resposta.

# Criando colunas para armazenar a variável resposta e as variáveis explicativas
Var_Explicativas = df.iloc[:,1:22]
print(Var_Explicativas.columns) # Verificando as colunas que estão armazenadas no objeto criado anteriormente
Var_Resposta = df['% Silica Concentrate']

# Criando o modelo de Regressão Linear Multivariada
import numpy as np
from sklearn.linear_model import LinearRegression
RegLinear_Inicial = LinearRegression().fit(Var_Explicativas, Var_Resposta) # Ajustando o modelo de Regressão Linear

# Obtendo os resultados do modelo
import statsmodels.api as sm
X2 = sm.add_constant(Var_Explicativas) # Adicionando uma constante a regressão (intercepto B0)
Reg_OLS = sm.OLS(Var_Resposta, X2) # Ajustando a regressão via OLS (Minimos Quadrados Ordinários, método classico)
Reg_OLS2 = Reg_OLS.fit() 
print(Reg_OLS2.summary()) # Saida com diveersas estatisticas com relação ao modelo

# Analisando problemas de multicolinearidade na Regressão criada
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
X = add_constant(Var_Explicativas) # termo relacionado a constante (ignorar no momento do VIF, mas importante para o calculo correto)
print(pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns))


### Construindo uma regressão linear multivariada considerando os indicadores de interesse ----

## Criando novamente os indicadores

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
                            
# Ajustando o modelo de regressão multivariado com indicadores
Var_Explicativas_ind = df.iloc[:,np.r_[1:8, 24:len(df.columns)]]
print(Var_Explicativas_ind.columns)

# Obtendo os resultados do modelo
X2 = sm.add_constant(Var_Explicativas_ind) # Adicionando uma constante a regressão (intercepto B0)
Reg_OLS_ind = sm.OLS(Var_Resposta, X2) # Ajustando a regressão via OLS 
Reg_OLS_ind2 = Reg_OLS_ind.fit() 
print(Reg_OLS_ind2.summary()) # Saida com diversas estatisticas relacionadas ao modelo

# Utilizando o método "Backward" para obter um modelo final apenas com variáveis significativas (valor-p < 0.050)
Var_Explicativas_atualizado = df.iloc[:,np.r_[2:8, 24:len(df.columns)]]
print(Var_Explicativas_atualizado.columns) # sem a variável "% Iron Feed"

# Resultados do modelo atualizado
X2 = sm.add_constant(Var_Explicativas_atualizado) # Adicionando uma constante a regressão (intercepto B0)
Reg_OLS_ind2 = sm.OLS(Var_Resposta, X2) # Ajustando a regressão via OLS 
Reg_OLS_ind3 = Reg_OLS_ind2.fit() 
print(Reg_OLS_ind3.summary()) # Resultados do modelo

# Verificando o VIF do modelo atualizado (sem "% Iron Feed" e com indicadores)
X = add_constant(Var_Explicativas_atualizado)
print(pd.Series([variance_inflation_factor(X.values, i) 
               for i in range(X.shape[1])], 
              index=X.columns))



### Analisando as suposições básicas da regressao linear multivariada: ----

## Análise de Resíduos
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Normalidade dos resíduos
Res_Normalidade = sns.distplot(Reg_OLS_ind3.resid, fit=stats.norm) # Curva de normalidade dos resíduos
Res_Normalidade.set(title = "Curva de Normalidade dos Resíduos da Regressão Multivariada")
plt.show()
sm.qqplot(Reg_OLS_ind3.resid, line='s') # QQPLOT dos resíduos
plt.show()

# Homocedasticidade dos residuos (gráfico Valores Preditos x Observados)
Y_min = Var_Resposta.min() # Valor minimo da diagonal do gráfico
Y_max = Var_Resposta.max() # Valor maximo da diagonal do gráfico
ax = sns.scatterplot(Reg_OLS_ind3.fittedvalues, Var_Resposta) # Criando um gráfico de pontos para os valores preditos e reais (observados)
ax.set(ylim=(Y_min, Y_max)) # Definindo os limites do eixo Y
ax.set(xlim=(Y_min, Y_max)) # Definindo os limites do eixo X
ax.set(title = "Gráfico de Homocedasticidade para os Resíduos da Regressão Multivariada")
ax.set_xlabel("Valor Predito para a variável resposta (% Silica Concentrate)")
ax.set_ylabel("Valor Observado para a variável resposta (% Silica Concentrate)")
X_ref = Y_ref = np.linspace(Y_min, Y_max, 100)  
plt.plot(X_ref, Y_ref, color='red', linewidth=1)
plt.show()

# Teste de Homocedasticidade dos resíduos de Breusch-Pagan
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
nomes = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value'] # Nomenado os resultados que serão calculados
test_BP = sms.het_breuschpagan(Reg_OLS_ind3.resid, Reg_OLS_ind3.model.exog) # Teste de Breusch-Pagan - H0: Homocedasticidade (Variância residual constante)
print(lzip(nomes, test_BP)) # Como o valor-p < 0.050, rejeitamos a hipóte nula de homocedasticidade dos dados 


### Conclusão Final: Como podemos ver, existem diversas violações nas suposições estatísticas básicas para utilização do modelo de regressão linear multivariado.
### Sendo assim, precisamos encontrar um modelo alternativo que não necessite de tais suposições paramétricas.



