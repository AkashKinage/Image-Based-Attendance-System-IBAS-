# Generated by Django 4.0.6 on 2022-10-08 09:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0002_attendance'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendance',
            name='attendance_image',
            field=models.ImageField(upload_to='Attendance_Images'),
        ),
    ]