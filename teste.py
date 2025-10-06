from psmpy import PsmPy
from psmpy.plotting import *

# Criar o objeto com os dados e as colunas necessárias (substitua conforme seu caso)
psm = PsmPy(your_dataframe, treatment='treatment_column', indx='index_column')

print(dir(psm))
