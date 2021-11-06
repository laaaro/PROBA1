import streamlit as st
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

header_container = st.container()
dataset = st.container()
features = st.container()
modelTraining = st.container()

##############################################################################

#AIXO MOLA MOOOLT EN CAS DE QUE TINGUI MOOOLTES DADES, AIXI NOMES HO CARREGO UNA VEGADA, ES GUARDA
# AL CACHÉ I AIXI NO HO HA DE TORNAR A CARREGAR TOT EL RATOO!! OLE OLEE LO CARACOLEE
@st.cache
def get_data(filename):
       taxi_data = pd.read_csv(filename)
       return taxi_data

##############################################################################

with header_container:
	st.title("WELCOMEEE!! OLE OLE LOS CARACOLEEES")
	st.write("Aquí mirare la base de dades de taxis a NY cityyyyy - amazinggg eh")


with dataset:
    st.header('NYC taxi dataset')
    st.text('I found this dataset on ....(mysteryy)')

 #LLEGIR CSV AMB PANDAS I VISUALITZAR TAULA
    taxi_data = get_data('DATA/taxi_data.csv') #si ho tinc a una altre carpeta ho he de dir, en aquest cas: DATA/
    st.write(taxi_data.head(4))   #EL HEAD ES PQ NOMES EM MOSTRI 40 FILES, I NO TOTA LA TAULA ENORME

 #GRAFICAR DADES DE LA TAULA
    st.subheader('Pick-up location ID distribution on the NYC dataset')
    PULocationID_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()) #value counts == conta el numero de vegades q es repeteix un numero
    st.bar_chart(PULocationID_dist.head(50)) # es a dir, cada ID fa referencia a un lloc, aleshores contem el numero de vegades q es recullen pedidos a un lloc
  

with features:
    st.header('The features I created')
    st.markdown('* **first feature: ** I created this feature because of .... and I calculated it using...')
    st.markdown('* **second feature: ** I created this feature because of .... and I calculated it using...')


with modelTraining:
    st.header('Time to train the model')
    st.subheader('Here u have to choose the parameters of the model and see the result!!')

 #ABANS HE DE CREAR 2 COLUMNES!!! PER ANAR POSANT LES COSES AL SEU LLOC
    sel_col , disp_col = st.columns(2)

 #PER AGAFAR INPUT NOMES HE DE FER:    variable = slider o el q siguiii!!!!
 ##slider / llista desplegable / mostro resultats / llista desplegable / text input  
    max_depth = sel_col.slider('What should be the max_depth of the model??', min_value=10 , max_value=100 , value=20 , step=10 ) # value == valor per defecte q es mostrara
    
    n_estimators = sel_col.selectbox('How many trees should there be??', options=[100,200,300,'No limitss alé alé'], index=0) #index=0 significa q per defecte mostrara el primer element de la llista
    
    sel_col.write('Here is a list of features in my data: ')
    sel_col.write(taxi_data.columns)

    sel_col.selectbox('Selecció de la llista :', options=taxi_data.columns, index=0) #index=0 significa q per defecte mostrara el primer element de la llista

    input_feature = sel_col.text_input('Which feature should be used as the input feature?', '-write here-')


    #disp_col.subheader('Mean absolute error of the model is:')
    #disp_col.subheader('Mean squared error of the model is:')
    #disp_col.subheader('R squared score of the model is:')
     


flag = True
if flag == False:
         

   #ARA CREO MODEL A ENTRENAR DE MAACHINE LEARNINGGG:
      regr = RandomForestRegressor(max_depth = max_depth , n_estimators = n_estimators)

      X = taxi_data[[input_feature]]
      y = taxi_data[['trip distance']]

      regr.fit(X,y)
      prediction = regr.predict(y)

      disp_col.subheader('Mean absolute error of the model is:')
      disp_col.write(mean_absolute_error(y,prediction))

      disp_col.subheader('Mean squared error of the model is:')
      disp_col.write(mean_squared_error(y,prediction))

      disp_col.subheader('R squared score of the model is:')
      disp_col.write(r2_score(y,prediction))




