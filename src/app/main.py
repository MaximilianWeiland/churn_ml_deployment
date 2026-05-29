from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import gradio as gr
from src.service.inference import predict

app = FastAPI(
    title="Netflix Customer Churn Prediction API",
    description="ML API for predicting customer churn for synthetic Netflix data",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}

class CustomerData(BaseModel):
    age: int
    gender: str
    subscription_type: str
    watch_hours: float
    last_login_days: int
    region: str
    device: str
    monthly_fee: float
    payment_method: str
    number_of_profiles: int
    avg_watch_time_per_day: float
    favorite_genre: str

@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def gradio_interface(
    age, gender, subscription_type, watch_hours, last_login_days, region,
    device, monthly_fee, payment_method, number_of_profiles,
    avg_watch_time_per_day, favorite_genre
):
    data = {
        "age": age,
        "gender": gender,
        "subscription_type": subscription_type,
        "watch_hours": watch_hours,
        "last_login_days": last_login_days,
        "region": region,
        "device": device,
        "monthly_fee": monthly_fee,
        "payment_method": payment_method,
        "number_of_profiles": number_of_profiles,
        "avg_watch_time_per_day": avg_watch_time_per_day,
        "favorite_genre": favorite_genre,
    }
    return str(predict(data))

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Number(label="Age", value=30, minimum=12, maximum=120),
        gr.Dropdown(["Male", "Female", "Other"], label="Gender", value="Male"),
        gr.Dropdown(["Basic", "Standard", "Premium"], label="Subscription Type", value="Basic"),
        gr.Number(label="Watch Hours", value=1, minimum=0),
        gr.Number(label="Last Login Days", value=1, minimum=1),
        gr.Dropdown(["Africa", "Europe", "Asia", "Oceania", "South America", "North America"], label="Region", value="Europe"),
        gr.Dropdown(["TV", "Mobile", "Laptop", "Desktop", "Tablet"], label="Device", value="Laptop"),
        gr.Number(label="Monthly Fee", value=13.99, minimum=0, maximum=20),
        gr.Dropdown(["Gift Card", "Crypto", "Debit Card", "PayPal", "Credit Card"], label="Payment Method", value="Credit Card"),
        gr.Number(label="Number of Profiles", value=1, minimum=0, maximum=5),
        gr.Number(label="Avg. Watch Time per Day", value=1, minimum=0, maximum=500),
        gr.Dropdown(["Action", "Sci-Fi", "Drama", "Horror", "Romance", "Comedy", "Documentary"], label="Favorite Genre", value="Action"),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="Netflix Customer Churn Predictor",
    description="Fill in the customer details below to get a churn prediction.",
    theme=gr.themes.Soft()
)

app = gr.mount_gradio_app(app, demo, path="/ui")
