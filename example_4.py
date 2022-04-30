import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

@st.cache
def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('data/occupancy_datatraining.txt', sep=",")
    return data


@st.cache
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    # Числовые колонки для масштабирования
    scale_cols = ['Temperature', 'Humidity', 'Light', 'CO2']
    new_cols = []
    sc1 = MinMaxScaler()
    sc1_data = sc1.fit_transform(data_out[scale_cols])
    for i in range(len(scale_cols)):
        col = scale_cols[i]
        new_col_name = col + '_scaled'
        new_cols.append(new_col_name)
        data_out[new_col_name] = sc1_data[:,i]

    temp_X = data_out[new_cols]
    temp_y = data_out['Occupancy']
    # Чтобы в тесте получилось низкое качество используем только 0,5% данных для обучения
    X_train, X_test, y_train, y_test = train_test_split(temp_X, temp_y, train_size=0.005, random_state=1)
    return X_train, X_test, y_train, y_test


# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, 
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")


# Модели
models_list = ['LogR', 'KNN_5', 'SVC', 'Tree', 'RF', 'GB']
clas_models = {'LogR': LogisticRegression(), 
               'KNN_5':KNeighborsClassifier(n_neighbors=5),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}


@st.cache(suppress_st_warning=True)
def print_models(models_select, X_train, X_test, y_train, y_test):
    current_models_list = []
    roc_auc_list = []
    for model_name in models_select:
        model = clas_models[model_name]
        model.fit(X_train, y_train)
        # Предсказание значений
        Y_pred = model.predict(X_test)
        # Предсказание вероятности класса "1" для roc auc
        Y_pred_proba_temp = model.predict_proba(X_test)
        Y_pred_proba = Y_pred_proba_temp[:,1]
   
        roc_auc = roc_auc_score(y_test.values, Y_pred_proba)
        current_models_list.append(model_name)
        roc_auc_list.append(roc_auc)

        #Отрисовка ROC-кривых 
        fig, ax = plt.subplots(ncols=2, figsize=(10,5))    
        draw_roc_curve(y_test.values, Y_pred_proba, ax[0])
        cm = confusion_matrix(y_test, Y_pred, normalize='all', labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax[1], cmap=plt.cm.Blues)
        fig.suptitle(model_name)
        st.pyplot(fig)

    if len(roc_auc_list)>0:
        temp_d = {'roc-auc': roc_auc_list}
        temp_df = pd.DataFrame(data=temp_d, index=current_models_list)
        st.bar_chart(temp_df)


st.sidebar.header('Модели машинного обучения')
models_select = st.sidebar.multiselect('Выберите модели', models_list)

data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)

st.header('Оценка качества моделей')
print_models(models_select, X_train, X_test, y_train, y_test)
