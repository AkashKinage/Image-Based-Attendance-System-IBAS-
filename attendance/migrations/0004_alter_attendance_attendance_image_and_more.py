# Generated by Django 4.0.6 on 2022-10-08 09:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0003_alter_attendance_attendance_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendance',
            name='attendance_image',
            field=models.ImageField(null=True, upload_to='Attendance_Images'),
        ),
        migrations.AlterField(
            model_name='attendance',
            name='date',
            field=models.DateField(null=True),
        ),
    ]