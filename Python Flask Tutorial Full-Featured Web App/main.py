from flask import Flask, render_template

app = Flask(__name__)

posts=[
    {
        "Author":"Amin",
        "Title":"Viva Palestina",
        "Content":"Palastine will be free",
        "Date_posted":"October, 7, 2023"
    },
    {
        
        "Author":"Mo",
        "Title":"Viva Palestina",
        "Content":"Palastine will be free eventually",
        "Date_posted":"October, 7, 2023"
    }
]

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)

@app.route("/about")
def about():
    return render_template('about.html',title=about)


if __name__=="__main__":
    app.run(debug=True)