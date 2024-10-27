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
        "spanish": "https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages/spanish.txt",
        #"russian": "https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages/russian.txt"
    }
    file_paths = {
        "english": "./words.txt",
        "french": "./french.txt",
        "italian": "./italian.txt",
        "spanish": "./spanish.txt",
        #"russian": "./russian.txt",

    }
    for language, url in word_urls.items():
        download_file(url, file_paths[language])


@app.route('/create_custom_dictionary', methods=['POST'])
def create_custom_dictionary():
    """
    Create a new custom Trie-based dictionary. This uses CUDA on the backend to speed up searches on large text corpuses. You will need to call Populate_Dictionary to store words. IMPORTANT: Make sure to destroy_custom_dictionary to free resources and stop charges.

    - **URL**: `/create_custom_dictionary`
    - **Method**: `POST`
    - **Request JSON**:
        - `dictionary` (str): The name of the custom dictionary.
        - `maxWordLength` (number): The maximum length of a word for this dictionary. Defaults to 20.
    - **Response JSON**:
        - `result` (str): status message.
    """
    data = request.get_json()
    dictionary = data.get("dictionary")
    maxWordLength = data.get("maxWordLength", 20)

    result = dictionary_service_instance.create_custom_dictionary(dictionary, maxWordLength)
    return jsonify({"result": "OK"})

@app.route('/destroy_custom_dictionary', methods=['POST'])
def destroy_custom_dictionary():
    """
    Destroys the dictionary from the server.

    - **URL**: `/destroy_custom_dictionary`
    - **Method**: `POST`
    - **Request JSON**:
        - `dictionary` (str): The name of the custom dictionary.
    - **Response JSON**:
        - `result` (str): status message.
    """
    data = request.get_json()
    dictionary = data.get("dictionary")

    result = dictionary_service_instance.destroy_custom_dictionary(dictionary)
    return jsonify({"result": "OK"})

@app.route('/populate_dictionary', methods=['POST'])
def populate_dictionary():
    """
    Add words to a custom dictionary. The wordlist should be a string containing space-separated words. Most unicode characters are allowed.

    - **URL**: `/destroy_custom_dictionary`
    - **Method**: `POST`
    - **Request JSON**:
        - `dictionary` (str): The name of the custom dictionary.
        - `wordlist` (str): lsit of words to store in the custom dictionary.
    - **Response JSON**:
        - `result` (str): status message.
    """
    data = request.get_json()
    dictionary = data.get("dictionary")
    wordlist = data.get("wordlist")
    result = dictionary_service_instance.populate_dictionary(dictionary, wordlist)
    return jsonify({"result": "OK"})

@app.route('/obtain_words', methods=['POST'])
def obtain_words():
    """
    Obtain words from a given text for use is populate dictionary. This returns a list of words in the form of a space-separated string.

    - **URL**: `/verify_words`
    - **Method**: `POST`
    - **Request JSON**:
        - `text` (str): The text to check.
    - **Response JSON**:
        - `result` (str): A space separated string of words..
    """
    data = request.get_json()
    text = data.get("text")

    result = dictionary_service_instance.obtain_words(text)
    return jsonify({"result": result})

@app.route('/similarity_check_of_two_dictionaries', methods=['POST'])
def similarity_check_of_two_dictionaries():
    """
    How similar are two dictionaries. Uses the formula log (common_words+1) / log (unique_words+1)

    - **URL**: `/calculate_text_integrity_from_bitmask`
    - **Method**: `POST`
    - **Request JSON**:
        - `dictionary1` (str): Name of the first dictionary
        - `dictionary2` (str): Name of the second dictionary
    - **Response JSON**:
        - `result` (number): The estimated percentage similarity of the two dictionaries.
    """
    data = request.get_json()
    dictionary1 = data.get("dictionary1")
    dictionary2 = data.get("dictionary2")
    result = dictionary_service_instance.similarity_check_of_two_dictionaries(dictionary1, dictionary2)
    return jsonify({"result": result * 100})

@app.route('/similarity_check_of_two_texts', methods=['POST'])
def similarity_check_of_two_texts():
    """
    How similar are two texts. Uses the formula log (common_characters+1) / log (unique_characters+1)

    - **URL**: `/calculate_text_integrity_from_bitmask`
    - **Method**: `POST`
    - **Request JSON**:
        - `text1` (str): Name of the first dictionary
        - `text2` (str): Name of the second dictionary
    - **Response JSON**:
        - `result` (number): The estimated percentage similarity of the two texts.
    """
    data = request.get_json()
    text1 = data.get("text1")
    text2 = data.get("text2")
    result = dictionary_service_instance.similarity_check_of_two_texts(text1, text2)
    return jsonify({"result": result * 100})

@app.route('/verify_words', methods=['POST'])
def verify_words():
    """
    Verify words in a given text for a specified language.

    - **URL**: `/verify_words`
    - **Method**: `POST`
    - **Request JSON**:
        - `text` (str): The text to check.
        - `language` (str): The language of the text. Supports "Russian", "English", "Spanish", "Italian" and "French".
    - **Response JSON**:
        - `result` (str): A CSV string of booleans indicating the presence of each word in the language dictionary.
    """
    data = request.get_json()
    text = data.get("text")
    language = data.get("language")

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
        - `text` (str): The text to filter.
        - `mask` (str): A CSV string of booleans indicating which words to keep.
    - **Response JSON**:
        - `result` (list): A list of filtered words that matched the boolean CSV string.
    """
    data = request.get_json()
    text = data.get("text")
    boolean_csv = data.get("mask")
    words = logic_operation_service_instance.filterWordsByBoolean(text, boolean_csv)
    return jsonify({"result": [str(word) for word in words]})

@app.route('/calculate_bitmask_relevance', methods=['POST'])
def calculate_bitmask_accuracy():
    """
    Calculates the number of 1's and divide that by the total number of elements. This estimates the percentage relevance of the bitmask.

    - **URL**: `/calculate_text_integrity_from_bitmask`
    - **Method**: `POST`
    - **Request JSON**:
        - `bitmask` (str): A string of comma separated ones and zeroes.
    - **Response JSON**:
        - `result` (number): The estimated percentage relevance of the bitmask.
    """
    data = request.get_json()
    bitmask = data.get("bitmask")
    result = logic_operation_service_instance.calculate_bitmask_relevance(bitmask)
    return jsonify({"result": result * 100})


if __name__ == '__main__':
    check_and_download_files()

    # Initialize services
    dictionary_service_instance = dictionary_service.DictionaryService.get_instance()
    logic_operation_service_instance = dictionary_service.LogicOperationService.get_instance()

    app.run(debug=False)
    