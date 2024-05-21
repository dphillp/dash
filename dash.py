import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


navegacao=['Portal','Dash','Previsao']
pagina=st.sidebar.selectbox('Navegacao',navegacao)
    
if pagina == 'Previsao':
    df = pd.read_csv('petroleo.csv')
    df.columns = ['ds','y']
    
    for lag in range(1, 2):
        df[f'Preco_lag_{lag}'] = df['y'].shift(lag)

    df=df.dropna()

    X = df[df['ds'] >= '2019-01-01']['Preco_lag_1'].values
    y = df[df['ds'] >= '2019-01-01']['y'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=False,random_state=42)

    X_train=X_train.reshape(-1,1)
    X_test=X_test.reshape(-1,1)
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)

    model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42,loss='squared_error')
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    
    previsao = pd.DataFrame(zip(df['ds'].iloc[-len(predictions):].values,predictions),columns=['data','preco_previsto'])
    
    st.markdown('#### Previsão')
    st.line_chart(data=previsao, x='data', y='preco_previsto', color='#FFA500', width=0, height=0, use_container_width=True)
    

elif pagina == 'Portal':
    st.markdown('#### Dados Originais')
    df = pd.read_csv('petroleo.csv')
    df.columns = ['Data','Valor']
    
    df['Data_aux'] = pd.to_datetime(df['Data'])
    df['Data'] = df['Data_aux'].dt.date
    df['Valor'] = df['Valor'].astype(float)
    df['Mes']  = df['Data_aux'].dt.month
    df['Ano']  = df['Data_aux'].dt.year
    df['Tri']  = df['Data_aux'].dt.quarter
    
    def mostra_qntd_linhas(dataframe):

        qntd_linhas = st.sidebar.slider('Selecione a quantidade de linhas que deseja mostrar na tabela', min_value = 10, max_value = len(dataframe), step = 1)
        st.write(dataframe.head(qntd_linhas).style.format(subset = ['Ano'], formatter="{:.2f}"))



# filtros para a tabela
    checkbox_mostrar_tabela = st.sidebar.checkbox('Mostrar tabela')
    if checkbox_mostrar_tabela:

        st.sidebar.markdown('## Filtro por ano')
        anos = list(df['Ano'].unique())
        anos.append('Todos')

        anual = st.sidebar.selectbox('Selecione um ano específico', options = anos)



        st.sidebar.markdown('## Filtro por trimestre')
        trimestres = [1,2,3,4]
        trimestres.append('Todos')

        trimestral = st.sidebar.selectbox('Selecione um trimestre específico', options = trimestres)

        if anual != 'Todos':
            if trimestral != 'Todos':
                mostra_qntd_linhas(df.query(f"Ano == {anual} and Tri == {trimestral}"))

            else:
                mostra_qntd_linhas(df.query(f"Ano =={anual}"))

        else:
            if trimestral != 'Todos':
                mostra_qntd_linhas(df.query(f"Tri == {trimestral}"))

            else:
                mostra_qntd_linhas(df)

else:
    
    df = pd.read_csv('petroleo.csv')
    df.columns = ['ds','y']
    
    for lag in range(1, 2):
        df[f'Preco_lag_{lag}'] = df['y'].shift(lag)

    df=df.dropna()

    X = df[df['ds'] >= '2019-01-01']['Preco_lag_1'].values
    y = df[df['ds'] >= '2019-01-01']['y'].values

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,shuffle=False,random_state=42)

    X_train=X_train.reshape(-1,1)
    X_test=X_test.reshape(-1,1)
    y_train=y_train.reshape(-1,1)
    y_test=y_test.reshape(-1,1)

    model = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=5,random_state=42,loss='squared_error')
    model.fit(X_train,y_train)

    predictions = model.predict(X_test)
    
    compara = pd.DataFrame(zip(df['ds'].iloc[-len(predictions):].values,y_test,predictions),columns=['data','preco_original','preco_previsto'])
    compara['data_aux'] = pd.to_datetime(compara['data'])
    compara['data'] = compara['data_aux'].dt.date
    compara['preco_original'] = compara['preco_original'].astype(float)
    compara['preco_previsto'] = compara['preco_previsto'].astype(float)
    compara['mes']  = compara['data_aux'].dt.month
    compara['ano']  = compara['data_aux'].dt.year
    
    group = compara.groupby(['ano', 'mes'], as_index = False)['preco_original'].agg([np.max,np.min])
    group.columns = ['ano','mes','maior_preco','menor_preco']
    
    compara = compara.merge(group,how='left',on=['ano','mes'])
    
    
    st.sidebar.markdown('## Filtro por ano')
    anos = list(compara['ano'].unique())
    
    anual = st.sidebar.selectbox('Selecione um ano específico', options = anos)
    
    st.line_chart(data=compara, x='data', y=['preco_original','preco_previsto'], color=['#FFA500','#D0D0F8'], width=0, height=0, use_container_width=True)
    
    
    df = pd.read_csv('petroleo.csv')
    df.columns = ['Data','Valor']

    df['Data_aux'] = pd.to_datetime(df['Data'])
    df['Data'] = df['Data_aux'].dt.date
    df['Valor'] = df['Valor'].astype(float)
    df['Ano_Mes'] = df['Data_aux'].dt.year.astype(str).str.cat(df['Data_aux'].dt.month.astype(str),sep='_')
    
    group = df.groupby('Ano_Mes', as_index = False)['Valor'].agg([np.max,np.min])
    group.columns = ['Ano_Mes','Maior','Menor']
    
    meses = ['2013_1','2013_2','2013_3','2013_4','2013_5','2013_6','2013_7','2013_8','2013_9','2013_10','2013_11','2013_12']
    
    
    st.bar_chart(data=group[group['Ano_Mes'].isin(meses)], x='Ano_Mes',y='Maior')
    st.bar_chart(data=compara[compara['ano']==anual], x='mes',y='maior_preco')
    
    
    
    
