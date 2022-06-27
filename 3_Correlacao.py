##### Carregando o banco de dados ----
import pandas as pd 

df = pd.read_csv(r'C:\Users\luish\Stefanini\Desafio IHM Stefanini\MiningProcess_Flotation_Plant_Database.csv',
                 parse_dates=['date'], # Definindo qual coluna apresenta dados tipo "data"
                 decimal=",") # Indicando o separados de decimais para evitar problemas futuros



##### Calculando correlações entre as variáveis numéricas do banco ----

### Correlações entre variáveis que apresentavam um mesmo conceito: Fluxo de Ar (7 colunas) e Nível de Espuma (7 colunas)

## Carregando os pacotes necessários
import numpy as np
import seaborn as sns # para os Heatmaps
import matplotlib.pyplot as plt
from scipy.stats import spearmanr # Calcula os valores-p das matrizes de correlação de Spearman

## Fluxo de Ar
Colunas_FluxoAr = df.iloc[:,8:15] 
pd.set_option('display.max_columns', None) # Esse argumento permite que todos as colunas apareçam nos outputs
Corr_FluxoAr = Colunas_FluxoAr.corr(method="spearman") # Utilizamos o método de Spearman pois é uma alternativa não-paramétrica ao de Pearson
print(np.around(Corr_FluxoAr,2)) # Visualizando a matriz com apenas 2 casas decimais
Corr_FluxoAr2, pvalor_Ar = spearmanr(Colunas_FluxoAr) # Obtendo separadamente os valores-p para as correlações de Spearman
print(np.around(pvalor_Ar,3)) # Valores-P da correlação de Spearman (todos significativos)

mask = np.triu(np.ones_like(Corr_FluxoAr, dtype=bool)) # Responsável por deixar apenas o triangulo inferior na plotagem
hm = sns.heatmap(Corr_FluxoAr, annot = True, mask = mask) # Heatmap para matriz de correlação
hm.set(title = "Correlação entre as variáveis relacionadas ao fluxo de ar das colunas de flotação")
sns.set(font_scale=0.8)
plt.show()

mask = np.triu(np.ones_like(pvalor_Ar, dtype=bool)) 
hm = sns.heatmap(pvalor_Ar, annot = True, mask = mask) 
hm.set(title = "P-valores das correlações entre as variáveis relacionadas ao fluxo de ar das colunas de flotação")
sns.set(font_scale=0.8)
plt.show()
# Conclusão: Todas as variáveis relacionadas ao fluxo de ar nas colunas da célula de flotação possuiram uma correlação estatisticamente significativa (valor-p < 0,050)


## Nível de Espuma
Colunas_NivelEspuma = df.iloc[:,15:22]  
print(Colunas_NivelEspuma)
Corr_NivelEspuma = Colunas_NivelEspuma.corr(method="spearman")
print(np.around(Corr_NivelEspuma,2)) 
Corr_NivelEspuma2, pvalor_NivelEspuma = spearmanr(Colunas_FluxoAr)
print(np.around(pvalor_NivelEspuma,3)) # Valores-P da correlação de Spearman (todos significativos)

mask = np.triu(np.ones_like(Corr_NivelEspuma, dtype=bool)) # Responsável por deixar apenas o triangulo inferior na plotagem
hm = sns.heatmap(Corr_NivelEspuma, annot = True, mask = mask) # Heatmap para matriz de correlação
hm.set(title = "Correlação entre as variáveis relacionadas ao nível de espuma das colunas de flotação")
sns.set(font_scale=0.8)
plt.show()

mask = np.triu(np.ones_like(pvalor_NivelEspuma, dtype=bool)) 
hm = sns.heatmap(pvalor_NivelEspuma, annot = True, mask = mask) 
hm.set(title = "P-valores das correlações entre as variáveis relacionadas ao nível de espuma das colunas de flotação")
sns.set(font_scale=0.8)
plt.show()
# Conclusão: Todas as variáveis relacionadas ao nível de espuma nas colunas da célula de flotação possuiram uma correlação estatisticamente significativa (valor-p < 0,050)

## Conclusão Final: Como essas variáveis eram correlacionadas entre si, seria interessante criar um indicador para cada uma delas, pois futuramente podiam causas problemas nos modelos (como no caso de Multicolinearidade).


### Criando Indicadores para as colunas relacionadas ao Fluxo de Ar e Nível de Espuma

# Fluxo de Ar
df['FluxoAr_indicador'] = df[['Flotation Column 01 Air Flow',
                              'Flotation Column 02 Air Flow',
                              'Flotation Column 03 Air Flow',
                              'Flotation Column 04 Air Flow',
                              'Flotation Column 05 Air Flow',
                              'Flotation Column 06 Air Flow',
                              'Flotation Column 07 Air Flow'
                            ]].mean(axis=1) # Calculando a média dessas colunas para cada linha ("axis=1")
print(df['FluxoAr_indicador'].head(5)) # Apresentando as 5 primeiras médias calculadas para o Fluxo de Ar

# Nível de Espuma
df['NivelEspuma_indicador'] = df[['Flotation Column 01 Level',
                                  'Flotation Column 02 Level',
                                  'Flotation Column 03 Level',
                                  'Flotation Column 04 Level',
                                  'Flotation Column 05 Level',
                                  'Flotation Column 06 Level',
                                  'Flotation Column 07 Level'
                                ]].mean(axis=1) 
print(df['NivelEspuma_indicador'].head(5)) 


### Correlação entre todas as variáveis numéricas (considerando os indicadores criados) do banco
Colunas_numericas = df.iloc[:,np.r_[1:8, 22:len(df.columns)]] # A função "np.r_" permite a concatenação de valores para a selação posterior de colunas
print(Colunas_numericas.head(5))
Corr_Todos = Colunas_numericas.corr(method="spearman")
print(np.around(Corr_Todos,2)) 
Corr_Todos2, pvalor_Todos = spearmanr(Colunas_numericas)
print(np.around(pvalor_Todos,3)) # Valores-P da correlação de Spearman (todos significativos)

mask = np.triu(np.ones_like(Corr_Todos, dtype=bool)) # Responsável por deixar apenas o triangulo inferior na plotagem
hm = sns.heatmap(Corr_Todos, annot = True, mask = mask) # Heatmap para matriz de correlação
hm.set(title = "Correlação entre as variáveis numéricas (incluindo indicadores criados) obtidas no banco de dados")
sns.set(font_scale=0.8)
plt.show()

mask = np.triu(np.ones_like(pvalor_Todos, dtype=bool)) # Responsável por deixar apenas o triangulo inferior na plotagem
hm = sns.heatmap(pvalor_Todos, annot = True, mask = mask) # Heatmap para matriz de correlação
hm.set(title = "P-valores das correlações entre as variáveis numéricas (incluindo indicadores criados) obtidas no banco de dados")
sns.set(font_scale=0.8)
plt.show()



##### Tentando aplicar a Análise de Componentes Principais (PCA) para estudar a correlação entre variáveis ----
from sklearn.decomposition import PCA

### Obtendo os autovetores e autovalores da matriz de correlação de Spearman
autovalores, autovetores = np.linalg.eig(Corr_Todos) # Obtendo os autovalores e autovetores que serão utilizados na PCA a partir da matriz de correlação de Spearman
print('Autovetores \n%s', np.sort(autovetores)) # Autovetores ordenados em ordem crescente
print('\n Autovalores (Variância Explicada por cada Componente) \n%s', np.sort(autovalores)[::-1]) # Autovalores ordenados em ordem decrescente (serão utilizados nas Componentes Principais geradas)

### Obtendo a variância explicada por cada Componente principal criada
tot = sum(autovalores) # Soma Total dos autovalores (Variância Explicada Total)
Var_explicada = [(i / tot) for i in sorted(autovalores, reverse=True)]  # Variância explicada por cada componente (autovalor obtido)
cum_var_exp = np.cumsum(Var_explicada) # Variância Explicada acumulada pelas "n" primeiras componentes principais (em ordem decrescente)
print('Proporção acumulada de variância explicada pelos primeiros "n" componentes principais: \n%s'  %cum_var_exp)

### Construindo um Scree-Plot para os calculos relacionados a PCA
plt.bar(range(1,12), Var_explicada, alpha=0.5, align='center', label='Variância Explicada por cada componente') # Criando as barras do gráfico relacionada a Var. Explicada por cada componente
plt.step(range(1,12), cum_var_exp, where= 'mid', label='Variância explicada acumulada') # Adicionando uma linha ao gráfico que mostra a variância acumulada explicada por "n" componentes
plt.ylabel('Proporção de variância explicada (%)') 
plt.xlabel('Componentes Principais')
plt.title('Gráfico de barras com linhas relacionadas as Componentes Principais geradas via PCA')
plt.legend(loc = 'best')
plt.show()

### Conclusão Final: Como as duas primeiras componentes principais geradas não conseguiram explicar ao menos 50% da variabilidade dos dados (37,9%), a utilização
### de mapas merceptuais para estudar a relação entre as variáveis passam a ser de pouca significância.
