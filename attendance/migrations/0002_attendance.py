# Generated by Django 4.0.6 on 2022-10-08 09:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Attendance',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('year', models.CharField(max_length=2)),
                ('division', models.CharField(max_length=1)),
                ('subject', models.CharField(max_length=10)),
                ('date', models.DateField()),
                ('attendance_image', models.ImageField(upload_to='')),
            ],
        ),
    ]
