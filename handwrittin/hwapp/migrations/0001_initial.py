# Generated by Django 3.2.16 on 2022-10-17 07:27

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PicUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('imagefield', models.ImageField(blank=True, upload_to='pic_upload')),
            ],
        ),
    ]
