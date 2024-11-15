from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    message = request.json['message']
    # reply = f"Received: {message}"  # Simulated response
    return jsonify({'reply': reply})

@app.route('/run_code', methods=['POST'])
def run_code():
    code = request.json['code']
    try:
        exec_globals = {}
        exec(code, exec_globals)
        output = exec_globals.get('output', 'Code executed successfully.')
    except Exception as e:
        output = str(e)
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True)
