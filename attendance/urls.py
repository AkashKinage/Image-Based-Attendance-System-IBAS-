from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('register_student/', views.register_student, name='register_student'),
    path('search_student/', views.search_student, name='search_student'),
    path('register_faces/', views.register_faces, name='register_faces'),
    path('upload_attendance/', views.upload_attendance, name='upload_attendance'),
    path('view_attendance/', views.view_attendance, name='view_attendance'),
    path('train_model/', views.train_model, name='train_model'),
]