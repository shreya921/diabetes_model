import os
import gradio as gr
import joblib

# Load the trained model
deployed_dt = joblib.load("diabetes_model.pkl")


# Prediction Function
def predict_diabetes(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree_function,
    age
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
        return "🩺 Prediction: High Risk of Diabetes (Positive)"
    else:
        return "✅ Prediction: Low Risk of Diabetes (Negative)"


# Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(label="Pregnancies (Number of times pregnant)"),
        gr.Number(label="Glucose"),
        gr.Number(label="Blood Pressure"),
        gr.Number(label="Skin Thickness"),
        gr.Number(label="Insulin"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age")
    ],

    outputs=gr.Text(label="Prediction"),

    title="🩺 Diabetes Prediction System",

    description="""
Enter the patient's medical information below to predict whether the person is likely to have diabetes using a trained Decision Tree Machine Learning model.
"""
)

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False
    )
