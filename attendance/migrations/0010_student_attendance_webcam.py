# Generated by Django 4.0.6 on 2022-11-06 10:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0009_student_attendance_roll_no'),
    ]

    operations = [
        migrations.CreateModel(
            name='Student_Attendance_Webcam',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('full_name', models.CharField(max_length=50)),
                ('year', models.CharField(max_length=2)),
                ('division', models.CharField(max_length=1)),
                ('subject', models.CharField(max_length=10, null=True)),
                ('date', models.DateField(null=True)),
                ('roll_no', models.IntegerField(default=0)),
            ],
        ),
    ]