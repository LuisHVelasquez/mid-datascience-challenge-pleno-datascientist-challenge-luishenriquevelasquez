##### Carregando o banco de dados ----
import pandas as pd 

df = pd.read_csv(r'C:\Users\luish\Stefanini\Desafio IHM Stefanini\MiningProcess_Flotation_Plant_Database.csv',
                 parse_dates=['date'], # Definindo qual coluna apresenta dados tipo "data"
                 decimal=",") # Indicando o separados de decimais para evitar problemas futuros



##### Análise de Normalidade da variáveis resposta (% Silica Concentrate) ----

### Separando a variável resposta de interesse
Var_resposta = df['% Silica Concentrate']


### Criando um histograma para visualizar normalidade
import matplotlib.pyplot as plt
import seaborn as sns # "python -m pip install seaborn": pacote responsável por criar gráficos de densidade

sns.distplot(a=Var_resposta) # Função responsável por criar o histograma (esperado um formato de "Sino" em caso de Normalidade)
plt.title('Gráfico de densidade para a porcentagem de Silica obtida ao final do processo de flotação') # Titulo do Gráfico
plt.xlabel('% de Silica obtida') # Título do Eixo X
plt.ylabel('Densidade') # Título do Eixo Y
plt.show()
# Conclusão: não obtemos o formato esperado de "sino"


### Curva de Normalidade (QQPlot ou Quantile-Quantile Plot)
import statsmodels.api as sm # "pip install statsmodels": utilizado para criar o gráfico QQPLOT

fig = sm.qqplot(Var_resposta, line='45') # Adicionando uma linha de 45 graus para a curva de normalidade desejada
plt.title('Gráfico de Normalidade Real (azul) e Esperado (vermelho)')
plt.show()
# Conclusão: Como os dados se afastam muito da curva de normalidade esperada (em vermelho), temos indicios da ausência de normalidade na variável resposta


### Teste de Normalidade de Anderson-Darling
from scipy.stats import anderson
print(anderson(Var_resposta)) # Valor obtido: 25946.21, Valor crítico esperado a 5% de significância: 0.736
# Conclusão: Como o valor obtido foi muito maior que o valor crítico esperado a 5% de significância, rejeitamos a hipótese nula de que os dados seguem uma distribuição Normal.

### Conclusão Final: A partir de todos essses testes, podemos concluir que a nossa variável resposta de interesse não segue uma Distruição Normal.
### Sendo assim, seria interessante utilzarmos técnicas estatísticas não-paramétricas em nosso projeto, ou seja, que não possuem a suposição básica de Dist. Normal.


