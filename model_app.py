from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, computed_field
import pickle
from constant import *
from typing import Annotated, Literal
import pandas as pd


# Importing Model
with open("model.pkl", "rb") as model:
    model = pickle.load(model)

# Pydantic model to validate incoming data

class UserInput(BaseModel):
    age:Annotated[int, Field(..., description="Age of User", lt=100)]
    weight:Annotated[float, Field(..., gt= 0, description="weight of User")]
    height:Annotated[float, Field(..., description="height of User",lt= 10)]
    income_lpa:Annotated[int, Field(..., description="Income of User", gt=0)]
    smoker:Annotated[bool, Field(..., description=" Is User is a smoker")]
    city:Annotated[str, Field(..., description="City of User", max_length=100)]
    occupation:Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'], Field(..., description='Occupation of the user')]

    @computed_field
    @property
    def bmi(self) -> float:
        return self.weight / (self.height * self.height)

    @computed_field
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker and self.bmi > 30:
            return "high"
        elif self.smoker or self.bmi > 20:
            return "middle"
        else:
            return "low"

    @computed_field
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle-aged"
        return "senior"

    @computed_field
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3


app = FastAPI()
@app.post("/predict")
def predict(user_input: UserInput):
    input_df = pd.DataFrame([{
        "bmi" : user_input.bmi,
        "life_style_risk" : user_input.lifestyle_risk,
        "age_group": user_input.age_group,
        "city_tier": user_input.city_tier,
        "income_lpa" : user_input.income_lpa,
        "occupation": user_input.occupation,
    }])
    prediction = model.predict(input_df)[0]
    return JSONResponse(status_code= 200, content={"predicted_category": prediction})
