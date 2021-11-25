from os import error
from flask import *
import numpy as np
import pickle 

app = Flask(__name__)
model= pickle.load(open('model.pk','rb'))


@app.route('/')
def form_page():
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