##### Carregando o banco de dados ----
import pandas as pd # Instalando: "pip install pandas"

df = pd.read_csv(r'C:\Users\luish\Stefanini\Desafio IHM Stefanini\MiningProcess_Flotation_Plant_Database.csv',
                 parse_dates=['date'], # Definindo qual coluna apresenta dados tipo "data"
                 decimal=",") # Indicando o separados de decimais para evitar problemas futuros
print(df.head(10))

# Analisando o tipo de cada coluna da base de dados carregada
Classes_colunas = df.dtypes
print(Classes_colunas) #  Uma data e todas as demais números com casas decimais



##### Análise Decritiva dos dados ----

### Análise de dados faltantes (NA's) do banco
Dados_NA = df.isna().sum() # Somando todos os casos (lógicos, TRUE/FALSE) que exisiam NA's em cada coluna
print(Dados_NA)


### Datas
data = df['date']
print(data.describe(datetime_is_numeric=False)) # 4097 tempos únicos de coleta


### Colunas Numéricas

# Informações medidas em "m³/h"
Colunas_m3h = df[["Starch Flow","Amina Flow"]]
print(Colunas_m3h.describe())

import matplotlib.pyplot as plt # Carregando o pacote necessário para se criar os box-plots
import numpy as np # pacote necessário para ordenar os nomes dos box-plots
plt.boxplot(Colunas_m3h) 
plt.title('Box-Plots comparando as variáveis relacionadas ao fluxo de algumas substâncias na célula de flotação') # Titulo do Gráfico
plt.xlabel('Substância') # Título do Eixo X
plt.ylabel('m³/h') # Título do Eixo Y
categorias_porc = ('Starch Flow','Amina Flow') # Nomes das categorias (variaveis) apresentadas na plotagem
x_pos = np.arange(len(categorias_porc)) + 1 # Somo um pois no Python a ordem começa pelo 0
plt.xticks(x_pos, categorias_porc)
plt.show() 


# Informações medidas em "t/h"
Coluna_th = df['Ore Pulp Flow']
print(Coluna_th.describe())


# Informações relacionada ao ph (0 a 14)
Coluna_ph = df['Ore Pulp pH']
print(Coluna_ph.describe())


# Informações medidas em "1 a 3 kg/cm³"
Coluna_kgcm3 = df['Ore Pulp Density']
print(Coluna_kgcm3.describe())


# Informações medidas em "Nm³/h" (Flotation Columns 1:7 Air Flow)
pd.set_option('display.max_columns', None) # Esse argumento permite que todos as colunas apareçam nos outputs
Colunas_nm3h = df.iloc[:,8:15] # A função "iloc" permite selecionarmos colunas pelas suas posições no banco de dados
print(Colunas_nm3h.describe())

fig = plt.figure(figsize =(10, 7)) # Editando o tamanho padrão que o gráfico é exibido
plt.boxplot(Colunas_nm3h) # Criando os box-plot's com base nas colunas de interesse
plt.title('Box-Plots comparando as variáveis relacionadas ao fluxo de ar nas colunas') # Titulo do Gráfico
plt.xlabel('Flotation Columns Air Flow') # Título do Eixo X
plt.ylabel('Nm³/h') # Título do Eixo Y
plt.show() # Apresentando o box plot


# Informações medidas em "mm (milimeters)" (Flotation Columns 1:7 Level)
Colunas_mm = df.iloc[:,15:22] 
print(Colunas_mm.describe())
plt.boxplot(Colunas_mm) 
plt.title('Box-Plots comparando as variáveis relacionadas ao nível de espuma nas colunas') # Titulo do Gráfico
plt.xlabel('Flotation Columns Level') # Título do Eixo X
plt.ylabel('mm (milimeters)') # Título do Eixo Y
plt.show() 

# Informações medidas em porcentagem (%)
Colunas_porc = df[['% Iron Feed','% Silica Feed','% Iron Concentrate',"% Silica Concentrate"]]
print(Colunas_porc.describe())

plt.boxplot(Colunas_porc) 
plt.title('Box-Plots comparando as variáveis relacionadas a presença de elementos (em %) nos minérios de ferros') # Titulo do Gráfico
plt.xlabel('Elementos presentes no minério') # Título do Eixo X
categorias_porc = ('% Iron Feed','% Silica Feed','% Iron Concentrate',"% Silica Concentrate")
x_pos = np.arange(len(categorias_porc)) + 1 # Somo um pois no Python a ordem começa pelo 0
plt.xticks(x_pos, categorias_porc)
plt.ylabel('%') # Título do Eixo Y
plt.show() 


