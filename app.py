from flask import Flask , render_template,request
import numpy as np
import pickle 
import pandas as pd
import sqlite3
import sqlalchemy
from premiere_ia import x_test
#from flask_mysqldb import MySQL


#parametre de connection MYSQL

app = Flask(__name__)
#app.config['MYSQL_HOST'] = 'localhost'
#app.config['MYSQL_USER'] = 'root'
#app.config['MYSQL_PASSWORD'] = 'Y@ungmind14'
#app.config['MYSQL_DB'] = 'test'

#mysql = MySQL(app)

model= pickle.load(open('model.pk','rb'))


@app.route('/')
def index():
    return render_template('homePage.html')


@app.route('/prediction_simple')
def form():
            return render_template('connection.html')


@app.route('/post_login',methods=['post','get'])
def post_page():
    for x in request.form.values():
        
        if not x :
            erreur= 'Veuillez Renseignez Tout les champs'
            return render_template('connection.html', erreur=erreur)
    
    
    recup_data=[ int(x) for x in request.form.values()]
    features = np.array(recup_data)
    features = features.reshape(1,-1)
    prediction = model.predict(features)
   

  
    return render_template('connection.html',predict=prediction)


@app.route('/predict_multiple',methods=['post','get'])
def predict_multiple():
    
    engine = sqlalchemy.create_engine('sqlite:///projCardiaque')
    conn = sqlite3.connect('projCardiaque')
    cur = conn.cursor()
    x_test.to_sql('x_test', engine, if_exists='replace',index=False)
    req= cur.execute('select * from x_test LIMIT 200 ').fetchall()
    ok  = [x for x in req]
    
    if request.method== 'POST':
        nbre = request.form['nbre']
        query = cur.execute(f'SELECT * FROM x_test ORDER BY Random() LIMIT {nbre}').fetchall()
        features = np.array(query)
        prediction = model.predict(features)
        return render_template('predict_multiple.html',nbre=nbre,data=ok,predict=prediction)
        
        
    

    return render_template('predict_multiple.html',data=ok)

@app.route('/redirect')
def redirect():
    return render_template('connection.html')



#@app.route('/test',methods=['get','post'])
#def login():
#    if request.method == 'GET':
#        return render_template('test.html')
#     
#   if request.method == 'POST':
#        name = request.form['nom']
#        prenoms = request.form['prenom']
#        age = request.form['age']
#        cursor = mysql.connection.cursor()
#        cursor.execute(''' INSERT INTO utisateur VALUES(%s,%s,%s,%s)''',(1,name,prenoms,age))
#        mysql.connection.commit()
#        cursor.close()
#       msg = "Informations enregistré avec succes"
#        return  render_template('test.html',succes=msg)
