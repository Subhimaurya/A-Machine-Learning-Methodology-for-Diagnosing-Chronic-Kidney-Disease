import os
import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pygal
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import mysql
from mysql.connector import cursor

mydb = mysql.connector.connect(host='localhost', user='root', password='', port='3307', database='kidney')

app = Flask(__name__)
app.config["upload folder"] = r'uploads'
global acc, x_train, y_train, x_test, y_test
df=pd.read_csv(r'C:\Users\LENOVO\Desktop\TK100232\DATA SET\kidney_disease.csv')
df1=df.copy()
print(df1.head())
# df["classification"] = df["classification"].astype(str).astype(int)

df['age'] = df['age'].fillna(55.0)
df['bp'] = df['bp'].fillna(80.0)
df['sg'] = df['sg'].fillna(1.017)
df['al'] = df['al'].fillna(0.0)
df['su'] = df['su'].fillna(0.0)
df['rbc'] = df['rbc'].fillna('Typical')
df['pc'] = df['pc'].fillna('expected')
df['pcc'] = df['pcc'].fillna('nearby')
df['ba'] = df['ba'].fillna('existing')
df['bgr'] = df['bgr'].fillna(121.0)
df['bu'] = df['bu'].fillna(42.0)
df['sc'] = df['sc'].fillna(3.072454)
df['sod'] = df['sod'].fillna(138.0)
df['pot'] = df['pot'].fillna(4.4)
df['hemo'] = df['hemo'].fillna(12.5)
df['wc'] = df['wc'].fillna(8000.0)
df['rc'] = df['rc'].fillna(4.8)
df['htn'] = df['htn'].fillna('absorving')
df['cad'] = df['cad'].fillna('absorving')
df['appet'] = df['appet'].fillna('fine')
df['pe'] = df['pe'].fillna('absorving')
df['ane'] = df['ane'].fillna('absorving')
data = {"rbc": {"normal": 0, "abnormal": 1, "Typical": 2}}
dataframe = df.replace(data, inplace=True)
data1 = {'pc': {"normal": 0, "abnormal": 1, "expected": 2}}

df2 = df.replace(data1, inplace=True)
data2 = {'pcc': {"present": 0, "notpresent": 1, "nearby": 2}}
df3 = df.replace(data2, inplace=True)
data3 = {'ba': {"present": 0, "notpresent": 1, "existing": 2}}
df4 = df.replace(data3, inplace=True)
data4 = {'htn': {"yes": 0, "no": 1, "absorving": 2}}
df5 = df.replace(data4, inplace=True)
df['pcv'] = df['pcv'].fillna(40.0)
data5 = {'cad': {"yes": 0, "no": 1, "absorving": 2}}
df6 = df.replace(data5, inplace=True)
data6 = {'appet': {"good": 0, "poor": 1, "fine": 2}}
df7 = df.replace(data6, inplace=True)
data7 = {'pe': {"yes": 0, "no": 1, "absorving": 2}}
df8 = df.replace(data7, inplace=True)
data8 = {'ane': {"yes": 0, "no": 1, "absorving": 2}}
df9 = df.replace(data8, inplace=True)
dataframe0 = df.drop(['id'], axis=1)
dataframe0['pcv'] = dataframe0['pcv'].fillna(40.0)






x = dataframe0.drop(['classification'], axis=1)
y = dataframe0['classification']
# x = df.iloc[:,:-1]
# y = df.iloc[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
print(x_train.columns)
print(x_test.shape)
print(y_test.shape)
print(x_train.shape)
print(y_train.shape)

@app.route('/')

def home():
    return render_template('index.html')

@app.route('/login',methods = ["POST","GET"])

def login():
    if request.method == "POST":
        email = request.form['email']
        password = request.form['password']
        sql = "SELECT * FROM ckd WHERE Email=%s and Password=%s"
        val = (email, password)
        cur = mydb.cursor()
        cur.execute(sql, val)
        results = cur.fetchall()
        mydb.commit()
        if len(results) >= 1:
            return render_template('loghome.html', msg='success')
        else:
            return render_template('login.html', msg='fail')

    return render_template('login.html')

@app.route('/loghome')
def a():
    return render_template('loghome.html')


@app.route('/Register',methods=['GET','POST'])
def Register():
    if request.method == "POST":
        name = request.form['name']
        email = request.form['email']
        psw = request.form['psw']
        cpsw = request.form['cpsw']
        if psw == cpsw:
            sql = 'SELECT * FROM ckd'
            cur = mydb.cursor()
            cur.execute(sql)
            all_emails = cur.fetchall()
            mydb.commit()
            all_emails = [i[2] for i in all_emails]
            if email in all_emails:
                return render_template('Register.html', msg='exists')
            else:
                sql = 'INSERT INTO ckd(name,Email,Password) values(%s,%s,%s)'
                cur = mydb.cursor()
                values = (name,email, psw)
                cur.execute(sql, values)
                mydb.commit()
                cur.close()
                return render_template('Register.html', msg='Success')
        else:
            return render_template('Register.html', msg='Mismatch')
    return render_template('Register.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload',methods = ["POST","GET"])
def upload():
    if request.method == "POST":
        file = request.files['file']
        filetype = os.path.splitext(file.filename)[1]
        if filetype == '.csv':
            path = os.path.join(app.config['upload folder'], file.filename)
            file.save(path)
            return render_template('load data.html', msg='success')
        else:
            return render_template('load data.html', msg='invalid')
    return render_template('load data.html')
@app.route('/view')
def view():

    global df
    file = os.listdir(app.config['upload folder'])
    path = os.path.join(app.config['upload folder'], file[0])
    df = pd.read_csv(path)
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(df)
    return render_template('view data.html', col_name=df.columns, row_val=list(df.values.tolist()))

@app.route('/model',methods = ["POST","GET"])
def model():
    if request.method == "POST":

        model = int(request.form['selected'])
        if model == 1:
            model = RandomForestClassifier(n_estimators = 10,random_state=10)
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            acc = accuracy_score(y_test,y_pred)
            acc1=acc.round(4)
            acc1=100*acc1
            print(acc)
            return render_template('model.html',msg = 'accuracy',ac=acc1)
        elif model == 2:
            model = SVC(kernel='linear', random_state=0)
            model.fit(x_train, y_train)
            svcc = model.predict(x_test)
            acc = accuracy_score(y_test,svcc)
            acc1=acc.round(4)
            acc1=100*acc1
            print(acc)
            return render_template('model.html',msg ='svmaccuracy', ac=acc1)
        elif model == 3:
            model = KNeighborsClassifier()
            model.fit(x_train, y_train)
            knnc = model.predict(x_test)
            acc = accuracy_score(y_test,knnc)
            acc1=acc.round(4)
            acc1=100*acc1
            print(acc)
            return render_template('model.html', msg = 'knnaccuracy',ac=acc1)
        elif model == 4:
            print('aaaaa')
            model = LogisticRegression()
            model.fit(x_train, y_train)
            lrc = model.predict(x_test)
            acc = accuracy_score(y_test, lrc)
            acc1 = acc.round(4)
            acc1 = 100 * acc1
            print(acc)
            return render_template('model.html', msg = 'logisticregessionaccuracy',ac=acc1)
        elif model == 5:
            model = GaussianNB()
            model.fit(x_train,y_train)
            nbc = model.predict(x_test)
            acc = accuracy_score(y_test,nbc)
            acc1=acc.round(4)
            acc1=100*acc1
            print(acc)
            return render_template('model.html', msg = 'NBaccuracy',ac=acc1)
        elif model == 6:
            # dataframe.replace({'ckd':0,'notckd':1},inplace=True)
            dataframe=df1
            encoder = LabelEncoder()
            dataframe['classification'] = encoder.fit_transform(dataframe['classification'])
            dataframe['rbc'] = encoder.fit_transform(dataframe['rbc'])
            dataframe['pc'] = encoder.fit_transform(dataframe['pc'])
            dataframe['pcc'] = encoder.fit_transform(dataframe['pcc'])
            dataframe['ba'] = encoder.fit_transform(dataframe['ba'])
            dataframe['htn'] = encoder.fit_transform(dataframe['htn'])
            dataframe['cad'] = encoder.fit_transform(dataframe['cad'])
            dataframe['appet'] = encoder.fit_transform(dataframe['appet'])
            dataframe['pe'] = encoder.fit_transform(dataframe['pe'])
            dataframe['ane'] = encoder.fit_transform(dataframe['ane'])

            dataframe['age'] = dataframe['age'].fillna(55.0)
            dataframe['bp'] = dataframe['bp'].fillna(80.0)
            dataframe['sg'] = dataframe['sg'].fillna(1.017)
            dataframe['al'] = dataframe['al'].fillna(0.0)
            dataframe['su'] = dataframe['su'].fillna(0.0)
            dataframe['rbc'] = dataframe['rbc'].fillna('Typical')
            dataframe['pc'] = dataframe['pc'].fillna('expected')
            dataframe['pcc'] = dataframe['pcc'].fillna('nearby')
            dataframe['ba'] = dataframe['ba'].fillna('existing')
            dataframe['bgr'] = dataframe['bgr'].fillna(121.0)
            dataframe['bu'] = dataframe['bu'].fillna(42.0)
            dataframe['sc'] = dataframe['sc'].fillna(3.072454)
            dataframe['sod'] = dataframe['sod'].fillna(138.0)
            dataframe['pot'] = dataframe['pot'].fillna(4.4)
            dataframe['hemo'] = dataframe['hemo'].fillna(12.5)
            dataframe['wc'] = dataframe['wc'].fillna(8000.0)
            dataframe['rc'] = dataframe['rc'].fillna(4.8)
            dataframe['pcv'] = dataframe['pcv'].fillna(40.8)

            a= dataframe.drop(['classification'], axis=1)
            b = dataframe['classification']

            a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=10)

            print('aaaaa')
            model = Sequential()
            model.add(Dense(30, activation='relu'))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(a_train, b_train, batch_size=500, epochs=20, validation_data=(a_test, b_test))
            abc = model.predict(a_test)
            print(abc)
            import numpy as np
            bcd = np.ravel([abc])
            print(bcd)
            xyz=np.round(bcd)
            acc = accuracy_score(b_test, xyz)
            print(acc)
            acc=acc*100
            acc=acc.round(4)

            # model = Sequential()
            # # model.layers(Dense(40,activation='relu',input_shape=y_test))
            # model.add(Dense(30, activation='relu'))
            # model.add(Dense(20, activation='relu'))
            # model.add(Dense(1, activation='sigmoid'))
            # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            # model.fit(x_train, y_train, batch_size=500, epochs=10,validation_data=(x_test, y_test))
            # model.evaluate(x_test, y_test)
            # annc=model.predict(x_test)
            # acc =accuracy_score(y_test,annc)
            # acc1=acc.round(4)
            # acc1=100*acc1
            return render_template('model.html', msg='acc',ac=acc)



    return render_template('model.html')

@app.route('/prediction',methods = ["POST","GET"])
def prediction():
    if request.method == 'POST':
        a = int(request.form['a'])
        b = int(request.form['b'])
        c = int(request.form['c'])
        d = int(request.form['d'])
        e = int(request.form['e'])
        f = int(request.form['f'])
        g = int(request.form['g'])
        h = int(request.form['h'])
        i = float(request.form['i'])
        j = int(request.form['j'])
        k = int(request.form['k'])
        l = float(request.form['l'])
        m = int(request.form['m'])
        n = int(request.form['n'])
        o = int(request.form['o'])
        p = int(request.form['p'])
        q = int(request.form['q'])

        s = int(request.form['s'])
        t = int(request.form['t'])
        u = int(request.form['u'])
        v = int(request.form['v'])
        w = int(request.form['w'])
        x = int(request.form['x'])
        val = [[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,s,t,u,v,w,x]]
        model = RandomForestClassifier()
        model.fit(x_train,y_train)
        y_pred = model.predict(val)
        print(y_pred)
        return render_template('prediction.html',result = y_pred,msg = 'success')
    return render_template('prediction.html')


@app.route('/charts',methods = ["POST","GET"])
def charts():
    pie_chart = pygal.Pie()
    pie_chart.title = 'Machine learning Methodology for diagnosing chronic disease'
    a =df[df['classification']=='ckd']
    pera = len(a) / df.shape[0]
    b = df[df['classification'] == 'notckd']
    perb = len(b)/df.shape[0]
    pie_chart.add('normal',perb)
    pie_chart.add('Abnormal',pera)
    pie_data=pie_chart.render_data_uri()
    return render_template('graph.html', pie_data=pie_data)


if __name__ == '__main__':
    app.run(debug=True)