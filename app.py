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
    #connection a la database
    engine = sqlalchemy.create_engine('sqlite:///projCardiaque')
    conn = sqlite3.connect('projCardiaque')
    cur = conn.cursor()
    #transfert du data frame dans notre database en supprimant la colone index
    x_test.to_sql('x_test', engine, if_exists='replace',index=False)
    # requete sql pour afficher 200 element de notre database sur la page 
    req= cur.execute('select * from x_test LIMIT 200 ').fetchall()
    #explorer la requete pour recuperer ligne par ligne afin de mieux l'afficher sur la page 
    ok  = [x for x in req]
    # traitement du formulaire qui ce trouve sur la page predict multiple , ce formulaire permet a lutilisateur d'entré un nombre 
    if request.method== 'POST':
        #recuperation du nobre 
        nbre = request.form['nbre']
        #requete pour renvoyer n elements aleatoire(n est le nombre choisir par lutilisateur)
        query = cur.execute(f'SELECT * FROM x_test ORDER BY Random() LIMIT {nbre}').fetchall()
        #convertir le resultas en tableau numpy pour permetre la prediction 
        features = np.array(query)
        #prediction 
        prediction = model.predict(features)
        return render_template('predict_multiple.html',nbre=nbre,data=ok,predict=prediction)
        
        
    

    return render_template('predict_multiple.html',data=ok)

#une route de redirection 
@app.route('/redirect')
def redirect():
    return render_template('connection.html')





#@app.route('/test',methods=['get','post'])
#def login():
#    if request.method == 'GET':
#        return render_template('test.html')
     
    if request.method == 'POST':
        name = request.form['nom']
        prenoms = request.form['prenom']
        age = request.form['age']
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO utisateur VALUES(%s,%s,%s,%s)''',(1,name,prenoms,age))
        mysql.connection.commit()
        cursor.close()
        msg = "Informations enregistré avec succes"
        return  render_template('test.html',succes=msg)
