"""
Minimal UI to invoke SMART
"""

from flask import Flask, request

from SMART_CyberBook import main
from SMART_CyberBook.parameters import get_smart_hyperparameters

app = Flask(__name__)

@app.route("/run-smart", methods=['POST'])
def index():
    data = request.get_json()
    course_id = data["course_id"]
    """
    fetch the (hard-coded) hyperparamters for SMART-CORE from a file.
    """
    smart_hyperparameters = get_smart_hyperparameters()
    main.main(course_id, smart_hyperparameters) # invoke SMART

    response = {}
    response['status'] = 200
    response['message'] = 'success'

    return (str(status))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
