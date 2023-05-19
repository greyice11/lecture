from flask import Flask, render_template
import mysql.connector
import matplotlib.pyplot as plt
import numpy as np
from flask import request,url_for,session, redirect
import bcrypt
import secrets
app= Flask(__name__)
app.secret_key = secrets.token_hex(16)
cnx = mysql.connector.connect(user='root', password='',
                              host='localhost', database='lecture')
cursor = cnx.cursor()
@app.route('/')
def home():
    return render_template('login.html')

@app.route('/users')
def users():
    query = "SELECT * FROM users2"
    cursor.execute(query)
    users = cursor.fetchall()

    names = [user[1] for user in users]
    ages = [user[3] for user in users]

    plt.scatter(names, ages)
    plt.xlabel('Name')  
    plt.ylabel('Age')
    plt.title('Age vs Name of Users')
    plt.xticks(rotation=90)

    plt.savefig('static/scatter.png')

    num_males = sum(user[4] == 'Male' for user in users)
    num_females = sum(user[4] == 'Female' for user in users)

    genders = ['Male', 'Female']
    counts = [num_males, num_females]
    plt.pie(counts, labels=genders)

    plt.savefig('static/gender_bar_chart.png')

    return render_template('users.html', users=users)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        query = "SELECT * FROM users3 WHERE username = %s"
        cursor.execute(query, (username,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode('utf-8'), user[3].encode('utf-8')):
            session['user_id'] = user[0]
            return redirect('/dashboard')
        else:
            error = "Invalid username or password"
            return render_template('login.html', error=error)
    else:
        return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')

        hashed_password = bcrypt.hashpw(password, bcrypt.gensalt())

        query = "INSERT INTO users3 (username, email, password) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, email, hashed_password))
        cnx.commit()

        return redirect('/login')
    else:
        return render_template('register.html')

@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')
    if not user_id:
        return redirect('/login')

    query = "SELECT * FROM users3 WHERE id = %s"
    cursor.execute(query, (user_id,))
    user = cursor.fetchone()
    print(user)
    return render_template('dashboard.html', user=user)

if __name__ == "__main__":
  app.run(debug=True)