
from flask import Flask, render_template, request, jsonify
from model import ResumeMatcher
from utils.parse_resume import extract_text_from_file

app = Flask(__name__)
matcher = ResumeMatcher(data_path="sample_data", use_precomputed=False)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/match", methods=["POST"])
def match():
    top_k = int(request.form.get("top_k", 5))
    job_text = request.form.get("job_text", "").strip()
    file = request.files.get("resume_file")
    resume_text = request.form.get("resume_text", "").strip()

    if file:
        raw_text = extract_text_from_file(file)
    else:
        raw_text = resume_text

    if not raw_text and not job_text:
        return jsonify({"error": "Provide either a resume or a job description"}), 400

    if raw_text and not job_text:
        results = matcher.match_resume_against_jobs(raw_text, top_k=top_k)
    else:
        results = matcher.match_job_against_resumes(job_text, top_k=top_k)

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
