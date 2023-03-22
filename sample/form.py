# importing Flask and other modules
from flask import Flask, request, render_template
from backend import best_college

 
# Flask constructor
app = Flask(__name__)  
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods =["GET", "POST"])
def form():
    if request.method == "POST":
        user_rank = request.form.get("rank")
        user_c = request.form.get("category")
        user_round = request.form.get("round")
        user_q= request.form.get("quota")
        user_p= request.form.get("pool")
      
        return best_college(user_rank,user_c,user_round,user_q,user_p)
    return render_template('form.html')

# @app.route('/temp')
# def temp():
#     return render_template('temp.html')
if __name__=='__main__':
   app.run()