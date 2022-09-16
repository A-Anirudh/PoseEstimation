from django.urls import path
from .views import *

urlpatterns = [
    path('', BlogListView.as_view(), name='BlogListView'),
    path('<slug:question>/', BlogDetailView.as_view(), name='BlogDetailView'),
]
