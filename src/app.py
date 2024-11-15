from flask import Flask, render_template, request, jsonify
import ollama


desiredModel = 'llama3.2:1b'
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_code', methods=['POST'])
def run_code():
    code = request.json['code']
    try:
        prompt = f"Tune the parameters in this code: {code}"
        response = ollama.chat(
            model=desiredModel,
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ]
        )
        ollamaResponse = response['message']['content']
        
        # Save response to a text file
        with open("Output.txt", "w", encoding="utf-8") as text_file:
            text_file.write(ollamaResponse)
        
        output = ollamaResponse
    except Exception as e:
        output = str(e)
    return jsonify({'output': output})


if __name__ == '__main__':
    app.run(debug=True)





# prompt = "how to solve quadratic equations?"

# response = ollama.chat(
#     model=desiredModel,
#     messages = [
#         {
#             'role':'user',
#             'content':prompt,
#         },
#     ])

# ollamaResponse = response['message']['content']

# print(ollamaResponse)

# with open("Output.txt", "w", encoding="utf-8") as text_file:
#     text_file.write(ollamaResponse)