from django.urls import path
from .views import *

urlpatterns = [
    path('', predictHome, name='predictHome'),
    path('result', predictionResult, name='predictionResult'),
]
