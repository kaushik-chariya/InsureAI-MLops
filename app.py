from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from typing import Optional

# =========================
# PROJECT IMPORTS....
# =========================
from src.constants import APP_HOST, APP_PORT

from src.pipline.prediction_pipeline import (
    VehicleData,
    VehicleDataClassifier
)

from src.pipline.training_pipeline import TrainPipeline


# =========================
# FASTAPI APP
# =========================
app = FastAPI()


# =========================
# STATIC FILES
# =========================
app.mount("/static", StaticFiles(directory="static"), name="static")


# =========================
# TEMPLATES
# =========================
templates = Jinja2Templates(directory="templates")


# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# FORM DATA CLASS
# =========================
class DataForm:

    def __init__(self, request: Request):

        self.request = request

        self.Gender = None
        self.Age = None
        self.Driving_License = None
        self.Region_Code = None
        self.Previously_Insured = None
        self.Annual_Premium = None
        self.Policy_Sales_Channel = None
        self.Vintage = None
        self.Vehicle_Age_lt_1_Year = None
        self.Vehicle_Age_gt_2_Years = None
        self.Vehicle_Damage_Yes = None

    async def get_vehicle_data(self):

        form = await self.request.form()

        self.Gender = int(form.get("Gender"))
        self.Age = int(form.get("Age"))
        self.Driving_License = int(form.get("Driving_License"))
        self.Region_Code = float(form.get("Region_Code"))
        self.Previously_Insured = int(form.get("Previously_Insured"))
        self.Annual_Premium = float(form.get("Annual_Premium"))
        self.Policy_Sales_Channel = float(form.get("Policy_Sales_Channel"))
        self.Vintage = int(form.get("Vintage"))
        self.Vehicle_Age_lt_1_Year = int(form.get("Vehicle_Age_lt_1_Year"))
        self.Vehicle_Age_gt_2_Years = int(form.get("Vehicle_Age_gt_2_Years"))
        self.Vehicle_Damage_Yes = int(form.get("Vehicle_Damage_Yes"))


# =========================
# HOME ROUTE
# =========================
@app.get("/")
async def index(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="vehicledata.html",
        context={
            "context": ""
        }
    )


# =========================
# TRAIN ROUTE
# =========================
@app.get("/train")
async def train_route():

    try:

        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return Response("Training Successful!")

    except Exception as e:

        return Response(f"Error Occurred: {e}")


# =========================
# PREDICTION ROUTE
# =========================
@app.post("/")
async def predict_route(request: Request):

    try:

        # =========================
        # GET FORM DATA
        # =========================
        form = DataForm(request)

        await form.get_vehicle_data()

        # =========================
        # CREATE INPUT OBJECT
        # =========================
        vehicle_data = VehicleData(
            Gender=form.Gender,
            Age=form.Age,
            Driving_License=form.Driving_License,
            Region_Code=form.Region_Code,
            Previously_Insured=form.Previously_Insured,
            Annual_Premium=form.Annual_Premium,
            Policy_Sales_Channel=form.Policy_Sales_Channel,
            Vintage=form.Vintage,
            Vehicle_Age_lt_1_Year=form.Vehicle_Age_lt_1_Year,
            Vehicle_Age_gt_2_Years=form.Vehicle_Age_gt_2_Years,
            Vehicle_Damage_Yes=form.Vehicle_Damage_Yes
        )

        # =========================
        # DATAFRAME
        # =========================
        vehicle_df = vehicle_data.get_vehicle_input_data_frame()

        # =========================
        # MODEL PREDICTION
        # =========================
        model_predictor = VehicleDataClassifier()

        prediction = model_predictor.predict(
            dataframe=vehicle_df
        )[0]

        # =========================
        # RESULT STATUS
        # =========================
        status = (
            "✅ Customer Interested in Vehicle Insurance"
            if prediction == 1
            else "❌ Customer Not Interested in Vehicle Insurance"
        )

        # =========================
        # RETURN RESULT
        # =========================
        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={
                "context": status
            }
        )

    except Exception as e:

        return templates.TemplateResponse(
            request=request,
            name="vehicledata.html",
            context={
                "context": f"Error: {str(e)}"
            }
        )


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    app_run(
        app,
        host=APP_HOST,
        port=APP_PORT
    )