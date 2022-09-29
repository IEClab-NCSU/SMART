"""
Run the Server...

"""

from flask import Flask, request, redirect, flash, render_template, Response
from createMapping import cluster_assessment_mapping
from preprocessing import *
import os

ALLOWED_EXTENSIONS = set(['txt'])

app = Flask(__name__)
global files

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/download1")
def downloadfile1():
    dirname = os.path.split(os.path.abspath(__file__))
    with open(dirname[0]+"/Results/assessment_skill.csv","r") as fp:
         csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=assessment_skill.csv"})

@app.route("/download2")
def downloadfile2():
    dirname = os.path.split(os.path.abspath(__file__))
    with open(dirname[0]+"/Results/text_skill.csv", "r") as fp:
         csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=text_skill.csv"})
'''
@app.route("/download3")
def downloadfile3():
    with open("Results/assessment_text.csv") as fp:
         csv = fp.read()
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition":
                 "attachment; filename=assessment_text.csv"})
'''


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    skills = []
    workbooks=[]
    assessments=[]
    if request.method == 'POST':
        files = request.files.getlist("file[]")

        if files[0].filename == '' or files[1].filename == '':
            fileerror = "No File Selected"
            return render_template('index.html', fileerror = fileerror)

        if not allowed_file(files[0].filename) or not allowed_file(files[1].filename):
            fileerror = "Only .txt format is allowed"
            return render_template('index.html', fileerror=fileerror)

        if files[0] and files[1]:
            workbook_file = files[0]
            assessment_file = files[1]
            workbook_content = workbook_file.read()
            assessment_content = assessment_file.read()
            clustering = request.form['clustering']
            num_clusters = request.form['num_clusters']
            num_iterations = request.form['num_iterations']

            if clustering == "Paragraph":
                use_sentence = False
            else:
                use_sentence = True

            workbooks = preprocess(workbook_content, use_sentence)
            assessments = preprocess(assessment_content, False)
            #print workbooks
            #print assessments
	    if num_clusters == "":
		    num_clusters = "50"
            
            if int(num_clusters) >= len(workbooks):
                error = 'Clusters cannot be equal to or greater than the number of text items'
                return render_template('index.html', error=error, w = len(workbooks), a=len(assessments))
            
            if num_iterations == "":
                num_iterations = "50000"
            
            # if num_clusters == "":
            #    num_clusters = "50"

            filename, skills = cluster_assessment_mapping(int(num_clusters), int(num_iterations),
                                                          workbooks, assessments)

    return render_template('index.html', skills = skills, w = len(workbooks), a=len(assessments))

if __name__ == '__main__':
    app.run(debug=True)
