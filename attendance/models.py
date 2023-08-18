from __future__ import division
from unittest.util import _MAX_LENGTH
from django.db import models

# Create your models here.
class Student(models.Model):
    full_name = models.CharField(max_length=50)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    roll_no = models.IntegerField()
    department = models.CharField(max_length=30)
    year = models.CharField(max_length=2)
    division = models.CharField(max_length=1)

    def __str__(self) -> str:
        return self.full_name

class Attendance(models.Model):
    year = models.CharField(max_length=2)
    division = models.CharField(max_length=1)
    subject = models.CharField(max_length=10, null=True)
    date = models.DateField(null=True)
    attendance_image = models.ImageField(upload_to='Attendance_Images/', null=True)

    def __str__(self) -> str:
        return '{0} {1} {2} {3}'.format(self.date,self.year,self.division,self.subject)

class Student_Attendance(models.Model):
    full_name = models.CharField(max_length=50)
    year = models.CharField(max_length=2)
    division = models.CharField(max_length=1)
    subject = models.CharField(max_length=10, null=True)
    date = models.DateField(null=True)
    roll_no = models.IntegerField(default=0)

    def __str__(self) -> str:
        return '{0} {1} {2} {3} {4}'.format(self.full_name,self.date,self.year,self.division,self.subject)

class Student_Attendance_Webcam(models.Model):
    full_name = models.CharField(max_length=50)
    year = models.CharField(max_length=2)
    division = models.CharField(max_length=1)
    subject = models.CharField(max_length=10, null=True)
    date = models.DateField(null=True)
    roll_no = models.IntegerField(default=0)

    def __str__(self) -> str:
        return '{0} {1} {2} {3} {4}'.format(self.full_name,self.date,self.year,self.division,self.subject)
