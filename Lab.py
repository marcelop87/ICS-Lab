import streamlit as st
import pandas as pd
import plotly_express as px
import numpy as np
#import datetime
#from PIL import Image


container = st.container()
col1,col2 = st.columns(2)



@st.cache_data
def load_data(file):

  sheets = ['ICS-2', 'ICS-X1', 'ICS-3', 'AQUIO-X1001', 'Sep. Entrada']
  dfl = []
  for sheet in sheets:
    dfm = pd.read_excel(file, sheet_name= sheet , header=1, usecols='A,F:O').drop([0,1,2], axis=0,)
    dfl.append(dfm)

  df= pd.concat(dfl, keys= sheets, names= ['Pozo'])
  df= df.replace({'\-' : np.nan , '\*' : np.nan, '^\s*$': np.nan}, regex=True)
  df.reset_index(inplace=True, level='Pozo')
  df.dropna(subset='Fecha', inplace=True)
  df.Fecha = df.Fecha.dt.date
  return df


def sort_data(df):
  st.sidebar.header("Data Filtering")

  # Sort Data
  sort_column = st.sidebar.selectbox("Sort by", df.columns)
  df = df.sort_values(by=sort_column)
  df.fillna()
  df.reset_index(inplace=True, drop=True)
  return df



#def group_by(df):
  #Group Data
  #group_column = st.sidebar.selectbox("Group by Sum",df.columns)
  #grouped_df = df.groupby(group_column).sum()
  #return grouped_df

#def group_by_mean(df):
  # Group Data
  #group_column = st.sidebar.selectbox("Group by Mean",df.columns)
  #grouped_df_mean = df.groupby(group_column).mean()
  #return grouped_df_mean 


def analyze_data(data):
  # Perform basic data analysis
  container.write(" # Data Analysis # ")
  container.write("Last Data Uploaded")
  container.write(data.set_index('Fecha').sort_index(ascending=False).head())
  #container.write("Description")
  #container.write(data.describe())
  #container.write("Data Corelation")
  #container.write(data.corr())
  #container.write("Data Rank")
  #container.write(data.rank())


  st.empty() 
  with col1:

    st.write("Columns Names ", data.columns)
  with col1:

    st.write("Columns Data Types: ", data.dtypes)

  with col2:
    st.write("Missing Values: ", data.isnull().sum())


  with col2:
    st.write("Unique Values: ", data.nunique())


  #with col2: 
    #st.write("standerd deviation:", data.std())


  sorted_df = sort_data(data)

  container.write("Sort Data")
  container.write(sorted_df)

  with col1:
    st.write("Number of rows: ", data.shape[0])

  with col1: 
    st.write("Number of columns: ", data.shape[1])

  #groupBySum = group_by(data)

  #container.write("Group by sum")
  #container.write(groupBySum)

  #groupByMean = group_by_mean(data)
  #container.write("Group by mean")
  #container.write(groupByMean)

def create_chart(chart_type, data, x_column, y_column):

  container.write(" # Data Visualization # ")
  if chart_type == "Bar":

    st.header("Bar Chart")
    fig = px.bar(data, x=x_column, y=y_column,color = 'Pozo')
    st.plotly_chart(fig)

  elif chart_type == "Line":
    st.header("Line Chart")
    fig = px.line(data, x=x_column, y=y_column, color='Pozo')
    st.plotly_chart(fig)

  elif chart_type == "Scatter":
    st.header("Scatter Chart")
    fig = px.scatter(data, x=x_column, y=y_column,color="Pozo")
    st.plotly_chart(fig)

  elif chart_type == "Histogram":
    st.header("Histogram Chart")

    fig = px.histogram(data, x=x_column, y=y_column,color = "Pozo",log_x = False,log_y = False)
    st.plotly_chart(fig)


def predict_values(data):

  st.title("Software for Prediction")
  st.write("""### We need some information to predict values""")
  datos = data
  sel_pozo = st.selectbox("Select Well", datos.Pozo.unique())
  sel_value = st.selectbox("Select Value to Predict", datos.columns[2:])
  #sel_date = st.date_input("Select the date to make the Prediction").toordinal()
  #ok = st.button("Calculate Prediction") 

  #if ok:
    #datos[datos['Pozo'] == sel_pozo]
    #datos.dropna(subset=[sel_value], inplace=True) 
    #datos.reset_index(inplace=True, drop=True)

    #X = datos.Fecha.apply(lambda x: x.toordinal()) 
    #y = datos[sel_value]

    #max_depth = [None, 2,4,6,8,10,12]
    #parameters = {"max_depth": max_depth}

    #regressor = DecisionTreeRegressor(random_state=0)
    #gs = GridSearchCV(regressor, parameters, scoring='neg_mean_squared_error')
    #gs.fit(X, y.values)

    #regressor = gs.best_estimator_

    #regressor.fit(X, y.values)

    #y_pred = regressor.predict(sel_date)
    #error = np.sqrt(mean_squared_error(y, y_pred))
    #print("${:,.02f}".format(error))

    #prediction = y_pred
    #st.subheader(f"The estimated prediction is ${prediction[0]:.2f}")


def main():
  #image = Image.open("pandasFuny.jpg")
  #container.image(image,width = 100)
  container.write(" # Data Analysis and Visualization # ")
  #st.sidebar.image(image,width = 50)
  file = st.sidebar.file_uploader("Upload a data set in CSV or EXCEL format", type=["csv","xlsx"])

  options = st.sidebar.radio('Pages',options = ['Data Analysis','Data visualization', 'Data Prediction'])

  if file is not None:
    data = load_data(file)

    if options == 'Data Analysis':
      analyze_data(data)

    if options =='Data visualization':

      #Create a sidebar for user options
      st.sidebar.title("Chart Options")

      chart_type = st.sidebar.selectbox("Select a chart type", ["Scatter", "Line", "Bar", "Histogram"])

      x_column = 'Fecha'

      y_column = st.sidebar.selectbox("Select the Y column", data.columns[2:])


      create_chart(chart_type, data, x_column, y_column)

    if options =='Data Prediction': 
      predict_values(data)



if __name__ == "__main__":
  main()
