# Generated by Django 4.0.6 on 2022-10-08 09:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0006_alter_attendance_attendance_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendance',
            name='attendance_image',
            field=models.ImageField(null=True, upload_to='media/Attendance_Images/'),
        ),
    ]
