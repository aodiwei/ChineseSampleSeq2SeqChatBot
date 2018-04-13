#!/usr/bin/env python3
# coding:utf-8
"""
__title__ = ''
__author__ = 'David Ao'
__mtime__ = '2018/4/11'
# 
"""
# !pip install - U - q PyDrive
import os


def install_pydrive():
    os.system('pip install - U - q PyDrive')


def download(dir_id):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials

    # 1. Authenticate and create the PyDrive client.
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(dir_id)}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))
        if file1['title'] == '03-Twitter-chatbot.py':
            continue
        idx_q = drive.CreateFile({'id': file1['id']})
        idx_q.GetContentFile(file1['title'])
