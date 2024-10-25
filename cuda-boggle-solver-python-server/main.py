from flask import Flask, request, jsonify
import os
import requests
import dictionary_service

app = Flask(__name__)


# Function to check and download dictionaries
def download_file(url, path):
    if not os.path.exists(path):
        response = requests.get(url)
        response.raise_for_status()
        with open(path, 'wb') as f:
            f.write(response.content)

# Verify dictionary files exist or download if missing
def check_and_download_files():
    word_urls = {
        "english": "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words.txt",
        "french": "https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages/french.txt",
        "italian": "https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages/italian.txt",
        "spanish": "https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages/spanish.txt"
    }
    file_paths = {
        "english": "./words.txt",
        "french": "./french.txt",
        "italian": "./italian.txt",
        "spanish": "./spanish.txt"
    }
    for language, url in word_urls.items():
        download_file(url, file_paths[language])



@app.route('/verify_words', methods=['POST'])
def verify_words():
    """
    Verify words in a given text for a specified language.

    - **URL**: `/verify_words`
    - **Method**: `POST`
    - **Request JSON**:
        - `input1` (str): The text to check.
        - `input2` (str): The language of the text (e.g., "english").
    - **Response JSON**:
        - `result` (str): A CSV string of booleans indicating the presence of each word in the language dictionary.
    """
    data = request.get_json()
    text = data.get("input1")
    language = data.get("input2")

    result = dictionary_service_instance.check_words(language, text)
    return jsonify({"result": result})

@app.route('/logical_not', methods=['POST'])
def logical_not():
    """
    Perform logical NOT on a CSV string of booleans.

    - **URL**: `/logical_not`
    - **Method**: `POST`
    - **Request JSON**:
        - `input1` (str): A CSV string of booleans.
    - **Response JSON**:
        - `result` (str): A CSV string of booleans, with each boolean negated.
    """
    data = request.get_json()
    input_csv = data.get("input1")
    result = logic_operation_service_instance.logicalNotString(input_csv)
    return jsonify({"result": result})

@app.route('/logical_or', methods=['POST'])
def logical_or():
    """
    Perform logical OR between two CSV strings of booleans.

    - **URL**: `/logical_or`
    - **Method**: `POST`
    - **Request JSON**:
        - `input1` (str): First CSV string of booleans.
        - `input2` (str): Second CSV string of booleans.
    - **Response JSON**:
        - `result` (str): A CSV string of booleans resulting from ORing the two input strings.
    """
    data = request.get_json()
    input1 = data.get("input1")
    input2 = data.get("input2")
    result = logic_operation_service_instance.logicalOrStrings(input1, input2)
    return jsonify({"result": result})

@app.route('/logical_and', methods=['POST'])
def logical_and():
    """
    Perform logical AND between two CSV strings of booleans.

    - **URL**: `/logical_and`
    - **Method**: `POST`
    - **Request JSON**:
        - `input1` (str): First CSV string of booleans.
        - `input2` (str): Second CSV string of booleans.
    - **Response JSON**:
        - `result` (str): A CSV string of booleans resulting from ANDing the two input strings.
    """
    data = request.get_json()
    input1 = data.get("input1")
    input2 = data.get("input2")
    result = logic_operation_service_instance.logicalAndStrings(input1, input2)
    return jsonify({"result": result})

@app.route('/logical_xor', methods=['POST'])
def logical_xor():
    """
    Perform logical XOR between two CSV strings of booleans.

    - **URL**: `/logical_xor`
    - **Method**: `POST`
    - **Request JSON**:
        - `input1` (str): First CSV string of booleans.
        - `input2` (str): Second CSV string of booleans.
    - **Response JSON**:
        - `result` (str): A CSV string of booleans resulting from XORing the two input strings.
    """
    data = request.get_json()
    input1 = data.get("input1")
    input2 = data.get("input2")
    result = logic_operation_service_instance.logicalXorStrings(input1, input2)
    return jsonify({"result": result})

@app.route('/filter_words', methods=['POST'])
def filter_words():
    """
    Filter words in a text based on a CSV boolean string.

    - **URL**: `/filter_words`
    - **Method**: `POST`
    - **Request JSON**:
        - `input1` (str): The text to filter.
        - `input2` (str): A CSV string of booleans indicating which words to keep.
    - **Response JSON**:
        - `result` (list): A list of filtered words that matched the boolean CSV string.
    """
    data = request.get_json()
    text = data.get("input1")
    boolean_csv = data.get("input2")
    words = logic_operation_service_instance.filterWordsByBoolean(text, boolean_csv)
    return jsonify({"result": [str(word) for word in words]})

if __name__ == '__main__':
    check_and_download_files()

    # Initialize services
    dictionary_service_instance = dictionary_service.DictionaryService.get_instance()
    logic_operation_service_instance = dictionary_service.LogicOperationService.get_instance()

    app.run(debug=False)