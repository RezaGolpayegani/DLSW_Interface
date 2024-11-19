from flask import Flask, render_template, request, jsonify
import ollama
import ast

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

        # Convert all of the lines that are not Python code to comments
        def comment_outside_code_blocks(input_string):
            lines = input_string.splitlines()
            commented_lines = []
            inside_code_block = False

            for line in lines:
                # Toggle `inside_code_block` state when encountering a line with ```
                if line.strip().startswith("```"):
                    inside_code_block = not inside_code_block
                    commented_lines.append(f"# {line}")  # Comment the backtick lines as well
                elif inside_code_block:
                    # Keep lines inside code blocks unchanged
                    commented_lines.append(line)
                else:
                    # Comment out lines outside code blocks
                    commented_lines.append(f"# {line}" if line.strip() else line)

            return "\n".join(commented_lines)

        # Example usage
        output = comment_outside_code_blocks(ollamaResponse)


        # Save response to a text file
        with open("Output.txt", "w", encoding="utf-8") as text_file:
            text_file.write(ollamaResponse)

    except Exception as e:
        output = str(e)
    return jsonify({'output': output})


if __name__ == '__main__':
    app.run(debug=True)
