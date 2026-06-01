import pandas as pd
import numpy as np
import joblib
import gradio as gr

# Load saved model and encoders
model = joblib.load("student_grade_model.pkl")
le_dict = joblib.load("label_encoders.pkl")

# Extract options for dropdowns
departments = list(le_dict["Department"].classes_)
activities = list(le_dict["Extracurricular_Activities"].classes_)
internet = list(le_dict["Internet_Access_at_Home"].classes_)

def predict_grade(Department, Midterm_Score, Assignments_Avg, Projects_Score,
                  Quizzes_Avg, Attendance, Study_Hours_per_Week,
                  Sleep_Hours_per_Night, Phone_Screen_Time,
                  Extracurricular_Activities, Internet_Access_at_Home):

    dept_enc = le_dict["Department"].transform([Department])[0]
    extra_enc = le_dict["Extracurricular_Activities"].transform([Extracurricular_Activities])[0]
    net_enc = le_dict["Internet_Access_at_Home"].transform([Internet_Access_at_Home])[0]

    data = pd.DataFrame([[dept_enc, Midterm_Score, Assignments_Avg, Projects_Score,
                          Quizzes_Avg, Attendance, Study_Hours_per_Week,
                          Sleep_Hours_per_Night, Phone_Screen_Time,
                          extra_enc, net_enc]],
                        columns=['Department', 'Midterm_Score', 'Assignments_Avg', 'Projects_Score',
                                 'Quizzes_Avg', 'Attendance (%)', 'Study_Hours_per_Week',
                                 'Sleep_Hours_per_Night', 'Phone_Screen_Time (hrs/day)',
                                 'Extracurricular_Activities', 'Internet_Access_at_Home'])

    pred = model.predict(data)
    grade = le_dict["Grade"].inverse_transform(pred)[0]
    probs = model.predict_proba(data)[0]
    confidence = round(np.max(probs) * 100, 2)

    feedback = []
    if Midterm_Score < 70:
        feedback.append("📘 Focus more on midterm preparation.")
    if Assignments_Avg < 70:
        feedback.append("✍️ Improve assignment submission and quality.")
    if Projects_Score < 70:
        feedback.append("💡 Work more on project execution.")
    if Quizzes_Avg < 70:
        feedback.append("🧠 Revise course material regularly.")
    if Attendance < 80:
        feedback.append("📅 Increase attendance for better consistency.")
    if Study_Hours_per_Week < 10:
        feedback.append("⏰ Study more hours per week.")
    if Sleep_Hours_per_Night < 6:
        feedback.append("😴 Maintain at least 6 hours of sleep.")
    if Phone_Screen_Time > 4:
        feedback.append("📱 Reduce screen time for better focus.")
    if Extracurricular_Activities == "No":
        feedback.append("🎯 Join extracurricular activities to enhance soft skills.")
    if Internet_Access_at_Home == "No":
        feedback.append("🌐 Ensure better internet access for study resources.")

    if len(feedback) < 3:
        feedback.extend([
            "✅ Keep up your strong academic habits!",
            "🌟 Continue to improve consistency and focus.",
            "📈 Track your learning progress regularly."
        ])

    return f"🎓 Predicted Grade: {grade}\nConfidence: {confidence}%", "\n".join(feedback[:3])


with gr.Blocks(css="""
body { background-color: #0e0a1a; color: #e0e0ff; font-family: 'Poppins', sans-serif; }
.gradio-container {
    background: linear-gradient(145deg, #141122, #1e1633);
    border: 1px solid #ff00ff33;
    border-radius: 12px;
    padding: 25px;
    box-shadow: 0 0 25px #9400ff55;
}
h1, h2, label { color: #ff66ff !important; text-shadow: 0 0 5px #ff00ff99; }
p { color: #00ffff; }
.gr-button {
    background: linear-gradient(90deg, #ff00cc, #00ffff);
    color: black !important;
    font-weight: bold;
    border-radius: 10px;
    padding: 12px 20px;
    box-shadow: 0 0 25px #00ffff88;
    font-size: 16px;
}
.gr-button:hover { box-shadow: 0 0 35px #ff00ffcc; transform: scale(1.05); }
.output-box textarea {
    background-color: #150f25 !important;
    border: 2px solid #ff00ff !important;
    border-radius: 10px !important;
    color: #00ffff !important;
    font-size: 16px !important;
    height: 180px !important;
    padding: 12px !important;
    box-shadow: 0px 0px 20px #00ffff55;
}
""") as demo:

    gr.Markdown("<h1 style='text-align:center;'>🎓 Student Performance Prediction</h1>")
    gr.Markdown("<p style='text-align:center;'>Predict student grades and receive personalized feedback instantly</p>")

    with gr.Row():
        with gr.Column():
            Department = gr.Dropdown(choices=departments, label="Department")
            Midterm_Score = gr.Slider(0, 100, value=75, label="Midterm Score")
            Assignments_Avg = gr.Slider(0, 100, value=80, label="Assignments Average")
            Projects_Score = gr.Slider(0, 100, value=85, label="Projects Score")
            Quizzes_Avg = gr.Slider(0, 100, value=78, label="Quizzes Average")
            Attendance = gr.Slider(0, 100, value=90, label="Attendance (%)")
            Study_Hours_per_Week = gr.Slider(0, 40, value=12, label="Study Hours per Week")
            Sleep_Hours_per_Night = gr.Slider(0, 12, value=7, label="Sleep Hours per Night")
            Phone_Screen_Time = gr.Slider(0, 10, value=3, label="Phone Screen Time (hrs/day)")
            Extracurricular_Activities = gr.Radio(choices=activities, label="Extracurricular Activities")
            Internet_Access_at_Home = gr.Radio(choices=internet, label="Internet Access at Home")

    submit_btn = gr.Button("🔮 Predict Grade")

    gr.Markdown("<hr style='border: 1px solid #ff00ff; margin: 25px 0;'>")
    gr.Markdown("<h2 style='text-align:center;'>📊 Prediction Results</h2>")

    with gr.Row():
        with gr.Column(scale=1):
            output_grade = gr.Textbox(label="🎯 Predicted Grade", lines=2, elem_classes=["output-box"])
        with gr.Column(scale=2):
            output_feedback = gr.Textbox(label="📋 Feedback", lines=6, elem_classes=["output-box"])

    submit_btn.click(
        fn=predict_grade,
        inputs=[Department, Midterm_Score, Assignments_Avg, Projects_Score,
                Quizzes_Avg, Attendance, Study_Hours_per_Week, Sleep_Hours_per_Night,
                Phone_Screen_Time, Extracurricular_Activities, Internet_Access_at_Home],
        outputs=[output_grade, output_feedback]
    )

demo.launch()
