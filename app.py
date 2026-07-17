
import os
import gradio as gr
import joblib

# Load model
deployed_dt = joblib.load("diabetes_model.pkl")


# Prediction function
def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age,
):

    input_data = [[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree_function,
        age
    ]]

    prediction = deployed_dt.predict(input_data)

    if prediction[0] == 1:
        return "🔴 High Risk of Diabetes (Positive)"
    else:
        return "🟢 Low Risk of Diabetes (Negative)"


interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Slider(0, 17, step=1, value=0,
                  label="🤰 Pregnancies"),

        gr.Slider(0, 200, step=1, value=120,
                  label="🩸 Glucose"),

        gr.Slider(0, 130, step=1, value=70,
                  label="❤️ Blood Pressure"),

        gr.Slider(0, 100, step=1, value=20,
                  label="📏 Skin Thickness"),

        gr.Slider(0, 900, step=1, value=80,
                  label="💉 Insulin"),

        gr.Slider(0, 70, step=0.1, value=25,
                  label="⚖️ BMI"),

        gr.Slider(0.0, 3.0, step=0.01, value=0.5,
                  label="🧬 Diabetes Pedigree Function"),

        gr.Slider(1, 100, step=1, value=30,
                  label="🎂 Age")
    ],

    outputs=gr.Textbox(label="Prediction"),

    title="🩺 Diabetes Prediction System",

    description="""
### 🩺 About this Project

This web application predicts whether a patient is likely to have diabetes using a Decision Tree Machine Learning model.

---

**👩‍💻 Developed By:** Shreya Goel

**🎓 Roll Number:** 28240023

**🏫 College:** Panipat Institute of Engineering and Technology (PIET)

**📚 Department:** B.Tech – Computer Science & Engineering (CSE)

**🔗 LinkedIn:**  
https://www.linkedin.com/in/shreya-goel-6b5a902bb
""",

    submit_btn="🔍 Predict Diabetes",
    clear_btn="🗑️ Clear",

    theme=gr.themes.Soft()
)


if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
