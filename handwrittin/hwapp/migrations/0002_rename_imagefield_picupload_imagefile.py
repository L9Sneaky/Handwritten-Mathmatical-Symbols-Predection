# Generated by Django 3.2.16 on 2022-10-17 07:58

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('hwapp', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='picupload',
            old_name='imagefield',
            new_name='imagefile',
        ),
    ]
