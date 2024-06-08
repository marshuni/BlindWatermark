from django.urls import path
from django.shortcuts import redirect
from . import views

urlpatterns = [
    path("", lambda request: redirect("index")),
    path('index/', views.index, name='index'),
    path('about/', views.about, name='about'),
]